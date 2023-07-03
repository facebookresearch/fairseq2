"""
Trains a Transformer model for Speech To Speech Machine Translation on UST dataset.

Example command:
fairseq2 train examples/cli/train_s2t.py -w /checkpoint/$USER/fairseq2/s2tt.eng-deu arch_name=tiny wandb_project=fairseq2_s2tt
"""
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
import torchtnt.utils
from torch import Tensor

import fairseq2.cli
import fairseq2.models.s2t_transformer as s2t
from fairseq2 import data
from fairseq2.cli.api import Env, Seq2Seq
from fairseq2.data import Collater, DataPipelineBuilder, StringLike
from fairseq2.data.text import MultilingualTokenizer, Tokenizer, VocabularyInfo
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.transformer import TransformerModel

try:
    import torchaudio  # type: ignore
except ImportError:
    print(
        "`train_s2t` requires torchaudio. You can install it with: `pip install torchaudio`"
    )
    raise


log = logging.getLogger(__name__)


task = Seq2Seq


DATADIR: str = "/checkpoint/guw/fairseq2/data/must-c-v1.0.eng-deu"


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


def train_spm_from_stream(
    cfg: TrainSpmConfig, stream: Iterable[str], output: Path
) -> None:
    try:
        tmp_path = Path(tempfile.mkstemp(suffix=".txt")[1])
        training_lines = cfg.training_lines
        with tmp_path.open("wt", encoding="utf-8") as tmp:
            for i, line in enumerate(stream):
                print(line, file=tmp)
                if i >= training_lines:
                    break
        train_spm(cfg, tmp_path, output)
    finally:
        tmp_path.unlink()


def tokenizer(
    env: Env,
    xp: fairseq2.cli.Xp,
    data_dir: str = DATADIR,
    lang: str = "eng_Latn",
    vocab_size: int = 1024 * 8,
) -> Tokenizer:
    spm_dir = xp.script.parent.parent / "spm"
    spm_dir.mkdir(parents=True, exist_ok=True)
    spm_path = spm_dir / f"{Path(DATADIR).name}.train_asr.{vocab_size}.spm"
    if not spm_path.exists():
        cfg = TrainSpmConfig(
            vocab_size=vocab_size,
            training_lines=1_000_000,
            control_tokens=[f"<audio:{lang}>", f"<lang:{lang}>"],
        )
        manifest_path = data_dir + "/train_asr.tsv"
        log.info(
            f"Will train {spm_path} with vocab_size={vocab_size} on {manifest_path}"
        )
        text_data = (
            data.text.read_text(str(manifest_path), rtrim=True)
            .skip(1)
            .map(lambda line: str(line).split("\t")[3])
            .and_return()
        )
        train_spm_from_stream(cfg, text_data, spm_path)
    else:
        log.info(f"Will reuse tokenizer from {spm_path}")

    return MultilingualTokenizer(
        spm_path.with_suffix(spm_path.suffix + ".model"),
        "asr",
        set(),
        {lang},
        lang,
        lang,
    )


def module(
    env: Env, s2t_transformer_config: s2t.S2TTransformerConfig, fsdp: bool = False
) -> TransformerModel:
    """The translation model, see transformer for configuration.

    - fsdp: enable FSDP (default is DDP when using several GPUs)
    """
    torchtnt.utils.seed(0)
    torch.cuda.manual_seed(0)

    model = s2t.create_s2t_transformer_model(s2t_transformer_config, env.device)
    if fsdp:
        from torch.distributed.fsdp.fully_sharded_data_parallel import (
            FullyShardedDataParallel as FSDP,
        )

        return FSDP(model)  # type: ignore[return-value]
    else:
        return fairseq2.cli.DDP(model, env)  # type: ignore[return-value]


def s2t_transformer_config(
    arch_name: str, vocab_info: VocabularyInfo
) -> s2t.S2TTransformerConfig:
    config = s2t.s2t_transformer_archs.get_config(arch_name)

    config.target_vocabulary_size = vocab_info.size
    config.target_pad_idx = vocab_info.pad_idx

    return config


def train_data(
    tokenizer: Tokenizer,
    env: Env,
    lang: str = "eng_Latn",
    batch_size: int = 16,
    data_dir: str = DATADIR,
) -> Iterable[Seq2SeqBatch]:
    return load_data_from_manifest(
        tokenizer,
        data_dir + "/train_asr.tsv",
        batch_size,
        lang,
        env,
    )


def valid_data(
    tokenizer: Tokenizer,
    env: Env,
    lang: str = "eng_Latn",
    batch_size: int = 16,
    data_dir: str = DATADIR,
) -> Iterable[Seq2SeqBatch]:
    return load_data_from_manifest(
        tokenizer, data_dir + "/dev_asr.tsv", batch_size, lang, env
    )


def _read_tsv_shard(manifest_path: str, env: Env) -> DataPipelineBuilder:
    return (
        data.text.read_text(manifest_path, rtrim=True)
        .skip(1)
        .shard(env.global_rank, env.world_size)
    )


def load_data_from_manifest(
    tokenizer: Tokenizer,
    manifest_path: str,
    batch_size: int,
    lang: str,
    env: Env,
) -> Iterable[Seq2SeqBatch]:
    num_parallel_calls = 8
    pad_idx = tokenizer.vocab_info.pad_idx
    src_audio_dataloader = (
        _read_tsv_shard(manifest_path, env)
        .map(_load_audio_feats, num_parallel_calls=num_parallel_calls)
        .bucket(batch_size)
        .map(Collater(pad_idx=0))
        .and_return()
    )
    src_n_frames_dataloader = (
        _read_tsv_shard(manifest_path, env)
        .map(lambda line: torch.tensor(int(str(line).split("\t")[2])))
        .bucket(batch_size)
        .map(Collater())
        .prefetch(1)
        .and_return()
    )
    tgt_text_dataloader = (
        _read_tsv_shard(manifest_path, env)
        .map(lambda line: str(line).split("\t")[3])
        .map(
            tokenizer.create_encoder(mode="target", lang=lang),
            num_parallel_calls=num_parallel_calls,
        )
        .bucket(batch_size)
        .map(Collater(pad_idx))
        .and_return()
    )

    def generate_batch(b: List[Tensor]) -> Seq2SeqBatch:
        target = b[2].to(device)

        target_mask = target.ne(pad_idx)

        target_lens = torch.count_nonzero(target_mask, dim=-1)

        return Seq2SeqBatch(
            # Move batch to gpu
            # TODO use a dedicated cuda stream
            source_seqs=b[0].to(device),
            source_seq_lens=b[1].to(device),
            target_seqs=target,
            target_seq_lens=target_lens,
        )

    device = env.device
    return (
        data.DataPipeline.zip(
            [src_audio_dataloader, src_n_frames_dataloader, tgt_text_dataloader]
        )
        .prefetch(8)
        .map(generate_batch)
        # Don't prefetch too much on the GPU
        .prefetch(1)
        .and_return()
    )  # type: ignore[no-any-return]


def _load_audio_feats(line: StringLike) -> Tensor:
    line = str(line)
    audio_desc = line.split("\t")[1]
    colon_parts = audio_desc.split(":")
    assert len(colon_parts) == 3
    wav = _load_audio_feats_byteoffset(colon_parts)
    # Drop channel dim
    return wav.squeeze(0)


def _load_audio_feats_byteoffset(parts: List[str]) -> Tensor:
    assert len(parts) == 3
    # We don't even need the length, it's part of the numpy header
    audio_path, byte_offset = parts[0], int(parts[1])
    with open(audio_path, "rb") as f:
        f.seek(byte_offset)
        magic_header = f.peek(8)[:8]
        # Handle precomputed audio features
        if magic_header == b"\x93NUMPY\x01\x00":
            return torch.from_numpy(np.load(f))
        wav, sample_rate = torchaudio.load(f)
        return wav  # type: ignore[no-any-return]


def vocab_info(tokenizer: Tokenizer) -> VocabularyInfo:
    """Cache metadata about the tokenizer"""
    log.info(f"vocab_info: {tokenizer.vocab_info}")
    return tokenizer.vocab_info


# This is important, it tells torch.hub how to reload our "task" which contains model and tokenizer.
fairseq2_hub = fairseq2.cli.fairseq2_hub

if __name__ == "__main__":
    import fairseq2.cli.commands

    fairseq2.cli.commands.main(__file__)
