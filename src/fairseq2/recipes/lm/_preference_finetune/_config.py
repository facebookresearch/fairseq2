# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed

from fairseq2.datasets.preference import (
    GENERIC_PREFERENCE_DATASET_FAMILY,
)
from fairseq2.optim import ADAMW_OPTIMIZER, AdamWConfig
from fairseq2.optim.lr_scheduler import COSINE_ANNEALING_LR, CosineAnnealingLRConfig
from fairseq2.recipes.config import (
    ActivationCheckpointingSection,
    CommonSection,
    DatasetSection,
    FSDPSection,
    GangSection,
    LRSchedulerSection,
    ModelSection,
    OptimizerSection,
    RegimeSection,
    TextTokenizerSection,
    TorchSection,
    TrainerSection,
)


@dataclass(kw_only=True)
class POFinetuneConfig:
    model: ModelSection = field(
        default_factory=lambda: ModelSection(name="llama3_1_8b_instruct")
    )

    dataset: POFinetuneDatasetSection = field(
        default_factory=lambda: POFinetuneDatasetSection()
    )

    tokenizer: TextTokenizerSection = field(
        default_factory=lambda: TextTokenizerSection(name="llama3_instruct")
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    trainer: TrainerSection = field(
        default_factory=lambda: TrainerSection(
            dtype=torch.bfloat16,
            data_parallelism="fsdp",
            fsdp=FSDPSection(fp32_reduce=True),
            activation_checkpointing=ActivationCheckpointingSection(mode="layerwise"),
        )
    )

    criterion: POCriterionSection

    optimizer: OptimizerSection = field(
        default_factory=lambda: OptimizerSection(
            name=ADAMW_OPTIMIZER,
            config=AdamWConfig(
                lr=5.5e-06, betas=(0.9, 0.95), weight_decay=0.1, impl="fused"
            ),
        )
    )

    lr_scheduler: LRSchedulerSection = field(
        default_factory=lambda: LRSchedulerSection(
            name=COSINE_ANNEALING_LR, config=CosineAnnealingLRConfig(final_lr_scale=0.2)
        )
    )

    regime: RegimeSection = field(
        default_factory=lambda: RegimeSection(
            num_steps=5000,
            checkpoint_every_n_steps=1000,
            keep_last_n_checkpoints=1,
            publish_metrics_every_n_steps=10,
        )
    )

    # The memory efficient SDPA implementation in PyTorch is numerically not
    # stable when used with padded inputs.
    common: CommonSection = field(
        default_factory=lambda: CommonSection(
            torch=TorchSection(default_sdpa="torch_math")
        )
    )


@dataclass(kw_only=True)
class POFinetuneDatasetSection(DatasetSection):
    name: str = "gsm8k_dpo"

    family: str = GENERIC_PREFERENCE_DATASET_FAMILY

    path: Path | None = None

    source_encode_mode: str = "prompt"
    """The encode mode for the prompt, determines what special tokens to add."""

    target_encode_mode: str = "prompt_response"
    """The encode mode for the target, determines what special tokens to add."""

    mask_source_tokens: bool = True
    """If ``False``, calculates loss on the `src` tokens as well as the `tgt` tokens."""

    min_seq_len: int = 1
    """The minimum sum of ``src + tgt_chosen`` and ``src + tgt_rejected``.
    Shorter sequences will be dropped."""

    max_seq_len: int = 8192
    """The maximum sum of ``src + tgt_chosen`` and ``src + tgt_rejected``.
    Longer sequences will be dropped."""

    max_num_tokens: int = 8192 * 2
    """The maximum number of total `src`, `tgt_chosen`, and `tgt_rejected` tokens per batch."""

    batch_size: int | None = None
    """If not ``None``, ignores `max_num_tokens` and each batch will have `batch_size` examples."""

    example_shuffle_window: int = 10_000
    """The size of the sliding window for shuffling examples."""

    batch_shuffle_window: int = 1_000
    """The size of the sliding window for shuffling batches."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    extras: dict[str, object] = field(default_factory=dict)
    """The dataset-specific extra options."""

    chat_mode: bool = False
    """If True, dataset jsonl must have 'chat' field with openai-like messages List[Dict] entries"""


@dataclass(kw_only=True)
class POCriterionSection:
    name: str

    config: object
