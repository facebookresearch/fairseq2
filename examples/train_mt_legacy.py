import functools
import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import torchtnt.framework
import torchtnt.framework.callbacks
import torchtnt.utils

import fairseq2.callbacks
import fairseq2.cli
import fairseq2.dataloader.huggingface
import fairseq2.dataloader.legacy
import fairseq2.distributed
import fairseq2.hub
import fairseq2.nn
import fairseq2.optim.lr_scheduler
from fairseq2.dataloader import Seq2SeqBatch
from fairseq2.distributed import Env
from fairseq2.generate.tokenizer import DictTokenizer
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


DATA_DIR = Path("/private/home/guw/github/fairseq/data-bin/iwslt14.tokenized")


def lang_pairs(langs: str) -> LangPairs:
    return [tuple(pair.split("-", 1)) for pair in langs.split(",")]  # type: ignore


def train_data(
    lang_pairs: LangPairs, env: Env, data_dir: Path = DATA_DIR
) -> Iterable[Seq2SeqBatch]:
    load_data = functools.partial(
        fairseq2.dataloader.legacy.BilingualDataloader,
        data_dir,
        device=env.device,
    )

    if len(lang_pairs) > 1:
        train: Iterable[Seq2SeqBatch] = fairseq2.dataloader.RoundRobin(
            [load_data(*pair, "train") for pair in lang_pairs],
        )
    else:
        train = load_data(*lang_pairs[0], "train")
    return train


def valid_data(
    lang_pairs: LangPairs, env: Env, data_dir: Path = DATA_DIR
) -> Iterable[Seq2SeqBatch]:
    return fairseq2.dataloader.legacy.BilingualDataloader(
        data_dir, *lang_pairs[0], "valid", device=env.device
    )


def tokenizer(
    env: Env, lang_pairs: LangPairs, data_dir: Path = DATA_DIR
) -> DictTokenizer:
    src_0 = lang_pairs[0][0]
    src_langs = set(pair[0] for pair in lang_pairs)
    tgt_langs = set(pair[1] for pair in lang_pairs)

    src_0 = lang_pairs[0][0]
    tokenizer = DictTokenizer.from_fairseq_dict_txt(data_dir / f"dict.{src_0}.txt")
    for lang in sorted(src_langs | tgt_langs):
        tokenizer.add_special_token(lang)
    return tokenizer


def model(env: Env, tokenizer: DictTokenizer) -> Transformer:
    cfg = TransformerConfig(
        src_num_tokens=tokenizer.vocab_size(),
        tgt_num_tokens=tokenizer.vocab_size(),
        src_padding_token_idx=tokenizer.PAD,
        tgt_padding_token_idx=tokenizer.PAD,
        num_enc_attn_heads=4,
        num_dec_attn_heads=4,
        ffn_inner_dim=1024,
        dropout_p=0,
    )

    torchtnt.utils.seed(1)
    torch.cuda.manual_seed(1)

    # Build on CPU then push to GPU. This allows to use the CPU RNG seed, like
    # fairseq1.
    return build_transformer(cfg).to(device=env.device)


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


hub_task = fairseq2.cli.hub_export(task, __file__)
