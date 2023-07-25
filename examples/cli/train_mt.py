"""
Trains a Transformer model for Machine Translation on the NLLB dataset.

Example command:
fairseq2 train examples/cli/train_mt.py -w /checkpoint/$USER/fairseq2/mt.cat_Latn-eng_Latn 'lang=cat_Latn-eng_Latn' wandb_project=fairseq2

fairseq2 evaluate -s /checkpoint/$USER/fairseq2/mt.cat_Latn-eng_Latn/epoch_0_step_10/

fairseq2 inference -s /checkpoint/$USER/fairseq2/mt.cat_Latn-eng_Latn/epoch_0_step_10/
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torchtnt.utils

import fairseq2.cli
from fairseq2.cli.api import Env, TranslationTask
from fairseq2.data import Collater
from fairseq2.data.text import MultilingualTokenizer, Tokenizer, VocabularyInfo
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.nllb import NllbConfig, create_nllb_model
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.typing import Device

try:
    import datasets  # type: ignore[import]
except ImportError:
    print(
        "`train_mt` requires HuggingFace Datasets library. You can install it with: `pip install datasets`"
    )
    raise

log = logging.getLogger(__name__)


LangPair = Tuple[str, str]


task = TranslationTask


class NllbDataLoader(Iterable[Seq2SeqBatch]):
    def __init__(
        self,
        src: str,
        tgt: str,
        split: str,
        tokenizer: Tokenizer,
        batch_size: int,
        env: Env,
    ):
        self.split = split
        self.src = src
        self.tgt = tgt
        self.batch_size = batch_size
        self.global_rank = env.global_rank
        self.world_size = env.world_size
        self.device = env.device
        self.epoch = 0

        if tokenizer:
            self.pad_idx = tokenizer.vocabulary_info.pad_idx

            tsk = "translation"

            self.src_encoder = tokenizer.create_encoder(
                tsk, lang=src, mode="source", pin_memory=True
            )
            self.tgt_encoder = tokenizer.create_encoder(
                tsk, lang=tgt, mode="target", pin_memory=True
            )

            self.collater = Collater(self.pad_idx)

        data: Mapping[str, datasets.Dataset] = {}
        assert split in ("train", "valid", "test")
        if split == "train":
            # try both language pairs
            try:
                data = datasets.load_dataset(
                    "allenai/nllb", f"{src}-{tgt}", save_infos=True, streaming=True
                )  # type: ignore
            except Exception:
                data = datasets.load_dataset(
                    "allenai/nllb", f"{tgt}-{src}", save_infos=True, streaming=True
                )  # type: ignore
            self.data = data["train"]
            self.extract_src = lambda sample: sample["translation"][src]
            self.extract_tgt = lambda sample: sample["translation"][tgt]
        else:
            flores_split = "dev" if split == "valid" else "devtest"
            data = datasets.load_dataset("facebook/flores", f"{src}-{tgt}")  # type: ignore
            self.data = data[flores_split]
            self.extract_src = lambda sample: sample["sentence_" + src]
            self.extract_tgt = lambda sample: sample["sentence_" + tgt]

        # HF doesn't allow to shard and stream at the same time ?
        # self.data = self.data.shard(num_shards=self.world_size, index=self.global_rank)

    def __iter__(self) -> Iterator[Seq2SeqBatch]:
        for src_sentences, tgt_sentences in self._iter_str():
            source_tokens = []
            target_tokens = []

            # TODO: expose parallel_for
            for s, t in zip(src_sentences, tgt_sentences):
                source_tokens.append(self.src_encoder(s))
                target_tokens.append(self.tgt_encoder(s))

            source = self.collater(source_tokens)
            target = self.collater(target_tokens)

            yield Seq2SeqBatch(
                source["seqs"], source["seq_lens"], target["seqs"], target["seq_lens"]
            )

    def _iter_str(self) -> Iterator[Tuple[Sequence[str], Sequence[str]]]:
        if hasattr(self.data, "__len__"):
            log.info(
                f"Starting {self.split} epoch {self.epoch} with {len(self.data)} samples from memory."
            )
        else:
            dataset_size = (
                getattr(self.data._info, "size_in_bytes", 0) / 1024 / 1024 / 1024
            )
            log.info(
                f"Starting {self.split} epoch {self.epoch} with {dataset_size:.1f}Gb of data, streaming from disk."
            )
        batch_src, batch_tgt = [], []
        for i, sample in enumerate(self.data):
            if i % self.world_size != self.global_rank:
                continue

            batch_src.append(self.extract_src(sample))
            batch_tgt.append(self.extract_tgt(sample))
            if len(batch_src) == self.batch_size:
                yield batch_src, batch_tgt
                batch_src, batch_tgt = [], []
        if batch_src:
            yield batch_src, batch_tgt

        log.info(f"End of epoch {self.epoch}, with {i * self.batch_size} samples")
        self.epoch += 1

    @staticmethod
    def combine_and_dump(
        src: str, tgt: str, split: str, output: Path, limit: int = 0
    ) -> None:
        env = Env(world_size=1, global_rank=0, device=Device("cpu"))
        loader = NllbDataLoader(
            src,
            tgt,
            split,
            tokenizer=None,  # type: ignore
            batch_size=16,
            env=env,
        )
        i = 0
        with output.open("wt") as o:
            for src_sentences, tgt_sentences in loader._iter_str():
                for src in src_sentences:
                    print(src, file=o)
                for tgt in tgt_sentences:
                    print(tgt, file=o)
                i += 2 * len(src_sentences)
                if 0 < limit < i:
                    break


@dataclass
class TrainSpmConfig:
    vocab_size: int = 2**16
    training_lines: int = 5_000_000
    seed_sentencepiece_size: int = 5_000_000
    character_coverage: float = 0.999995
    model_type: str = "unigram"
    shuffle_input_sentence: bool = True
    num_threads: int = 4
    pad_idx: int = 0
    unk_idx: int = 1
    bos_idx: int = 2
    eos_idx: int = 3
    control_tokens: Optional[List[str]] = None


def train_spm(cfg: TrainSpmConfig, text_file: Path, output: Path) -> None:
    try:
        import sentencepiece
    except ImportError:
        print(
            "`train_mt` requires sentencepiece library. You can install it with: `pip install sentencepiece`"
        )
        raise

    spm_output = Path(str(output) + ".model")
    assert not output.exists(), f"Sentencepiece training would override {output}"
    assert (
        not spm_output.exists()
    ), f"Sentencepiece training would override {spm_output}"
    log.info(f"Training sentencepiece model on: {text_file}, output model: {output}")

    if cfg.control_tokens:
        user_defined_symbols = ",".join(cfg.control_tokens)
    else:
        user_defined_symbols = None

    sentencepiece.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=str(output),
        vocab_size=int(cfg.vocab_size),
        character_coverage=cfg.character_coverage,
        model_type=cfg.model_type,
        input_sentence_size=cfg.training_lines,
        seed_sentencepiece_size=min(cfg.training_lines, cfg.seed_sentencepiece_size),
        shuffle_input_sentence=cfg.shuffle_input_sentence,
        num_threads=cfg.num_threads,
        pad_id=cfg.pad_idx,
        unk_id=cfg.unk_idx,
        bos_id=cfg.bos_idx,
        eos_id=cfg.eos_idx,
        user_defined_symbols=user_defined_symbols,
    )

    spm_output.rename(output)
    log.info("sentencepiece training completed.")


def lang_pair(lang: str) -> LangPair:
    """Lang pair to train on."""
    src_lang, tgt_lang = lang.split("-", maxsplit=1)

    return src_lang, tgt_lang


def tokenizer(
    xp: fairseq2.cli.Xp, lang_pair: LangPair, spm_path: Optional[Path] = None
) -> Tokenizer:
    """Tokenizer

    - spm_path: path to a pretrained SentencePiece model. A new SPM will be trained if not given.
    """
    workdir = xp.script.parent
    if spm_path is not None:
        assert spm_path.exists(), f"Spm not found: {spm_path}"
    else:
        # TODO: this is a bit problematic because we are using the filesystem as a cache.
        # Inference model need this path to exists even though the SPM is also copied
        # in the task state.
        spm_path = workdir / "sentencepiece.model"

    src_lang, tgt_lang = lang_pair

    if not spm_path.exists():
        lang_tokens = [f"<lang:{lang}>" for lang in [src_lang, tgt_lang]]

        spm_train_txt = workdir / "spm_train_combined.txt"
        cfg = TrainSpmConfig(
            vocab_size=2**16,
            training_lines=1_000_000,
            control_tokens=lang_tokens,
        )
        NllbDataLoader.combine_and_dump(
            src_lang, tgt_lang, "train", spm_train_txt, limit=cfg.training_lines
        )
        train_spm(cfg, spm_train_txt, spm_path)
        assert spm_path.exists()

    workdir_spm = workdir / spm_path.name
    if workdir_spm.exists():
        if workdir_spm.resolve() != spm_path.resolve():
            raise Exception(
                f"Can't override existing spm in {workdir_spm}. Chose a new workdir or manually remove the previous spm."
            )
    else:
        workdir_spm.symlink_to(spm_path.resolve())

    task = "translation"

    return MultilingualTokenizer(
        spm_path, task, {src_lang}, {tgt_lang}, src_lang, tgt_lang
    )


def train_data(
    tokenizer: Tokenizer, env: Env, lang_pair: LangPair, batch_size: int = 16
) -> Iterable[Seq2SeqBatch]:
    src_lang, tgt_lang = lang_pair

    return NllbDataLoader(
        src_lang,
        tgt_lang,
        tokenizer=tokenizer,
        batch_size=batch_size,
        env=env,
        split="train",
    )


def valid_data(
    tokenizer: Tokenizer, env: Env, lang_pair: LangPair, batch_size: int = 16
) -> Iterable[Seq2SeqBatch]:
    src_lang, tgt_lang = lang_pair

    return NllbDataLoader(
        src_lang,
        tgt_lang,
        tokenizer=tokenizer,
        batch_size=batch_size,
        env=env,
        split="valid",
    )


def vocab_info(tokenizer: Tokenizer) -> VocabularyInfo:
    """Cache metadata about the tokenizer"""
    return tokenizer.vocabulary_info


def module(env: Env, model_config: NllbConfig) -> EncoderDecoderModel:
    """The translation model, see model_config for configuration"""
    torchtnt.utils.seed(0)
    torch.cuda.manual_seed(0)
    return create_nllb_model(model_config, env.device)


# Override default values of NllbConfig
def model_config(vocab_info: VocabularyInfo) -> NllbConfig:
    return NllbConfig(
        model_dim=512,
        max_seq_len=1024,
        vocabulary_size=vocab_info.size,
        pad_idx=vocab_info.pad_idx,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_encoder_attn_heads=16,
        num_decoder_attn_heads=16,
        ffn_inner_dim=512,
        dropout_p=0,
    )


# This is important, it tells torch.hub how to reload our "task" which contains model and tokenizer.
fairseq2_hub = fairseq2.cli.fairseq2_hub

if __name__ == "__main__":
    import fairseq2.cli.commands

    fairseq2.cli.commands.main(__file__)
