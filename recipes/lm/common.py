# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias
from fairseq2.datasets import SyncMode, SequenceBatch


import torch
from torch import Tensor

from fairseq2.logging import log
from fairseq2.models.clm import CausalLM
from fairseq2.nn import Embedding
from fairseq2.recipe import RecipeContext
from fairseq2.recipe.error import RecipeError


def check_model_vocabulary(context: RecipeContext) -> None:
    model = context.get_model_as(CausalLM)

    embed = _maybe_get_embed(model)
    if embed is None:
        raise RecipeError("Model must be a text-only causal language model.")

    tokenizer = context.get_tokenizer()

    vocab_info = tokenizer.vocab_info

    if embed.num_embeddings > vocab_info.size:
        log.warning("Vocabulary size of the tokenizer ({}) is less than the number of embeddings in the model ({}). If this is not intentional (e.g. padding for efficient GPU utilization), check your job configuration.", vocab_info.size, embed.num_embeddings)  # fmt: skip

    if embed.num_embeddings < vocab_info.size:
        raise RecipeError(
            f"Number of embeddings in the model ({embed.num_embeddings}) is less than the vocabulary size of the tokenizer ({vocab_info.size})."
        )

    if embed.pad_idx != vocab_info.pad_idx:
        raise RecipeError(
            f"Padding index in the embedding table ({embed.pad_idx}) does not match the pad token index in the tokenizer ({vocab_info.pad_idx})."
        )


def _maybe_get_embed(model: CausalLM) -> Embedding | None:
    try:
        embed = model.decoder_frontend.embed  # type: ignore[union-attr]
    except AttributeError:
        return None

    if not isinstance(embed, Embedding):
        return None

    return embed


def _gather_lprobs_avg(logits: Tensor, target: SequenceBatch) -> tuple[Tensor, Tensor]:
    assert target.target_mask is not None
    logprobs = torch.log_softmax(logits, dim=-1)
    per_token_logps = torch.gather(logprobs, -1, target.seqs.unsqueeze(-1)).squeeze(-1)
    total_logps = (per_token_logps * target.target_mask).sum(dim=-1)  # [Batch, 1]
    assert target.target_mask is not None
    average_logps = total_logps / target.target_mask.sum(-1)

    return total_logps, average_logps


@dataclass
class StaticBatching:
    """Specifies batching where each batch has the same number of examples."""

    batch_size: int
    """The number of examples in each batch."""


@dataclass
class LengthBatching:
    """Specifies batching where each batch has a maximum number of elements."""

    max_num_elements: int
    """The maximum number of elements (e.g. tokens) in each batch."""


Batching: TypeAlias = StaticBatching | LengthBatching


@dataclass(kw_only=True)
class DataReadOptions:
    batching: Batching = field(default_factory=lambda: StaticBatching(1))
    """The batching strategy for returned examples."""

    example_shuffle_window: int = 0
    """
    The size of the sliding window for shuffling examples. If ``1``, no
    shuffling is performed; if ``0``, true shuffling is performed by loading the
    entire dataset.
    """

    batch_shuffle_window: int = 0
    """
    The size of the sliding window for shuffling batches. If ``1``, no
    shuffling is performed; if ``0``, true shuffling is performed by loading the
    entire dataset.
    """

    drop_remainder: bool = False
    """
    If ``True``, drops the last set of batches if they have in total fewer
    examples than requested.
    """

    sync_batches: bool = True
    """
    If ``True``, ensures that each process in the gang reads the same number of
    batches. Typically used when the amount of data to be read can vary per
    process (e.g. due to unbalanced sharding or non-static batching) and it is
    critical for each process to iterate over the same number of batches (e.g.
    during training).
    """

    sync_mode: SyncMode = SyncMode.UNTIL_FIRST
    """
    The data synchronization mode among processes in the gang. Only effective if
    :attr:`sync_batches` is ``True``.
    """

    max_num_batches: int | None = None
    """The maximum number of batches to return."""

    num_accumulate: int = 1
    """
    The number of batches to accumulate in each iteration. Typically used with
    gradient accumulation during training.
    """

    prefetch: int = 1
    """The number of batches to prefetch in background."""

    npc: int = 10
    """The reference number of parallel calls that data reader can do."""

    seed: int = 2
    """The seed to initialize the random number generators used internally."""
