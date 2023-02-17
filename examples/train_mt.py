"""
Trains a Transformer model for Machine Translation using the NLLB dataset on HuggingFace.

Example command:
python -m fairseq2 train examples/train_mt.py -w /checkpoint/$USER/fairseq2/mt.cat_Latn-eng_Latn 'langs=cat_Latn-eng_Latn' --wandb=fairseq2

python -m fairseq2 evaluate -s /checkpoint/$USER/fairseq2/mt.cat_Latn-eng_Latn/epoch_0_step_10/

python -m fairseq2 inference -s /checkpoint/$USER/fairseq2/mt.cat_Latn-eng_Latn/epoch_0_step_10/
"""
import functools
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torchtnt.utils

import fairseq2.callbacks
import fairseq2.cli
import fairseq2.dataloader.huggingface
import fairseq2.distributed
import fairseq2.nn
import fairseq2.optim.lr_scheduler
from fairseq2.dataloader import Seq2SeqBatch
from fairseq2.distributed import Env
from fairseq2.generate import SpmTokenizer, TokenMeta, spm_train
from fairseq2.models.transformer import (
    Transformer,
    TransformerConfig,
    build_transformer,
)
from fairseq2.optim.lr_scheduler import InverseSquareRootLR, LRScheduler
from fairseq2.tasks import TranslationTask

log = logging.getLogger(__name__)

LangPairs = List[Tuple[str, str]]


task = TranslationTask


def lang_pairs(langs: str) -> LangPairs:
    return [tuple(pair.split("-", 1)) for pair in langs.split(",")]  # type: ignore


def tokenizer(
    env: Env, lang_pairs: LangPairs, spm_path: Optional[Path] = None
) -> SpmTokenizer:
    workdir = env.workdir
    if spm_path is not None:
        assert spm_path.exists(), f"Spm not found: {spm_path}"
    else:
        # TODO: this is a bit problematic because we are using the filesystem as a cache.
        # Inference model need this path to exists even though the SPM is also copied
        # in the task state.
        spm_path = workdir / "sentencepiece.model"

    if not spm_path.exists():
        spm_train_txt = workdir / "spm_train_combined.txt"
        # TODO handle more language pairs
        src, tgt = lang_pairs[0]
        cfg = spm_train.TrainSpmConfig(
            vocab_size=2**16 - 300,  # 2 ** 16 - some room for special tokens
            training_lines=1_000_000,
        )
        fairseq2.dataloader.huggingface.NllbDataLoader.combine_and_dump(
            src,
            tgt,
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

    _tokenizer = SpmTokenizer.from_file(spm_path)
    src_langs = set(pair[0] for pair in lang_pairs)
    tgt_langs = set(pair[1] for pair in lang_pairs)
    lang_tokens = {}
    for lang in sorted(src_langs | tgt_langs):
        lang_tokens[lang] = _tokenizer.add_special_token(lang)

    return _tokenizer


def train_data(
    tokenizer: SpmTokenizer, env: Env, lang_pairs: LangPairs, batch_size: int = 16
) -> Iterable[Seq2SeqBatch]:
    load_data = functools.partial(
        fairseq2.dataloader.huggingface.NllbDataLoader,
        tokenizer=tokenizer,
        batch_size=batch_size,
        env=env,
        split="train",
    )

    return fairseq2.dataloader.RoundRobin([load_data(*pair) for pair in lang_pairs])


def valid_data(
    tokenizer: SpmTokenizer, env: Env, lang_pairs: LangPairs, batch_size: int = 16
) -> Iterable[Seq2SeqBatch]:
    return fairseq2.dataloader.huggingface.NllbDataLoader(
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
# Similarly we should export the Builder config in the config.
def token_meta(tokenizer: SpmTokenizer) -> TokenMeta:
    return TokenMeta.from_tokenizer(tokenizer)


def model(env: Env, token_meta: TokenMeta) -> Transformer:
    torchtnt.utils.seed(0)
    torch.cuda.manual_seed(0)

    # TODO: this is problematic for inference because we force the spm path to exists on disk
    # to create the model, while we want to load it from the snapshot.
    # How can we create the model without the tokenizer ?
    cfg = TransformerConfig(
        src_num_tokens=token_meta.vocab_size,
        tgt_num_tokens=token_meta.vocab_size,
        src_padding_token_idx=token_meta.PAD,
        tgt_padding_token_idx=token_meta.PAD,
        dropout_p=0,
    )

    return build_transformer(cfg, env.device)


def optimizer(model: Transformer, weight_decay: float = 0.001) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.0001,
    )


def lr_scheduler(optimizer: torch.optim.Optimizer) -> LRScheduler:
    return InverseSquareRootLR(optimizer, lr=5e-4)


# TODO: should we append this when generating hubconf.py ?
hub_task = fairseq2.cli.hub_export(task, __file__)
