"""
Trains a Transformer model for Machine Translation on the NLLB dataset.

Example command:
fairseq2 train examples/train_mt.py -w /checkpoint/$USER/fairseq2/mt.cat_Latn-eng_Latn 'langs=cat_Latn-eng_Latn' wandb_project=fairseq2

fairseq2 evaluate -s /checkpoint/$USER/fairseq2/mt.cat_Latn-eng_Latn/epoch_0_step_10/

fairseq2 inference -s /checkpoint/$USER/fairseq2/mt.cat_Latn-eng_Latn/epoch_0_step_10/
"""
import functools
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torchtnt.utils

import fairseq2.cli
import fairseq2.data.huggingface
from fairseq2.cli import Env
from fairseq2.cli.api import TranslationTask
from fairseq2.data import Seq2SeqBatch
from fairseq2.data.text import MultilingualTokenizer, Tokenizer, VocabularyInfo
from fairseq2.generate import spm_train
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.nllb import NllbConfig, create_nllb_model

log = logging.getLogger(__name__)

LangPairs = List[Tuple[str, str]]


task = TranslationTask


def lang_pairs(langs: str) -> LangPairs:
    """List of lang pairs to train on.

    - langs: comma separated list of lang pairs, eg: "eng_Latn-fra_Latn,eng_Latn-oci_Latn"
    """
    return [tuple(pair.split("-", 1)) for pair in langs.split(",")]  # type: ignore


def tokenizer(
    xp: fairseq2.cli.Xp, lang_pairs: LangPairs, spm_path: Optional[Path] = None
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

    src_langs = set(pair[0] for pair in lang_pairs)
    tgt_langs = set(pair[1] for pair in lang_pairs)

    default_src_lang, default_tgt_lang = lang_pairs[0]

    if not spm_path.exists():
        lang_tokens = [f"<lang:{lang}>" for lang in sorted(src_langs | tgt_langs)]

        spm_train_txt = workdir / "spm_train_combined.txt"
        # TODO handle more language pairs
        cfg = spm_train.TrainSpmConfig(
            vocab_size=2**16,
            training_lines=1_000_000,
            control_tokens=lang_tokens,
        )
        fairseq2.data.huggingface.NllbDataLoader.combine_and_dump(
            default_src_lang,
            default_tgt_lang,
            "train",
            spm_train_txt,
            limit=cfg.training_lines,
        )
        spm_train.train(cfg, spm_train_txt, spm_path)
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
        spm_path, task, src_langs, tgt_langs, default_src_lang, default_tgt_lang
    )


def train_data(
    tokenizer: Tokenizer, env: Env, lang_pairs: LangPairs, batch_size: int = 16
) -> Iterable[Seq2SeqBatch]:
    load_data = functools.partial(
        fairseq2.data.huggingface.NllbDataLoader,
        tokenizer=tokenizer,
        batch_size=batch_size,
        env=env,
        split="train",
    )

    return fairseq2.data.huggingface.RoundRobin(
        [load_data(*pair) for pair in lang_pairs]
    )


def valid_data(
    tokenizer: Tokenizer, env: Env, lang_pairs: LangPairs, batch_size: int = 16
) -> Iterable[Seq2SeqBatch]:
    return fairseq2.data.huggingface.NllbDataLoader(
        *lang_pairs[0],
        tokenizer=tokenizer,
        batch_size=batch_size,
        env=env,
        split="valid",
    )


# TODO: I'm not really happy to have this in user code.
# the tokenizer is serialized with the model, but we still need to know the vocab_size
# to create an embedding matrix of the right size.
# Should we hide that in cli.py@train ?
# Similarly we should export the model config in the config.
def vocab_info(tokenizer: Tokenizer) -> VocabularyInfo:
    """Cache metadata about the tokenizer"""
    return tokenizer.vocab_info


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
