# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

from fairseq2.recipe.config import (
    ADAMW_OPTIMIZER,
    COSINE_ANNEALING_LR,
    AdamWConfig,
    CommonSection,
    CompileOptions,
    CosineAnnealingLRConfig,
    DatasetSection,
    GangSection,
    LRSchedulerSection,
    ModelSection,
    OptimizerSection,
    RegimeSection,
    TokenizerSection,
    TorchConfig,
    TrainerSection,
)

from .dataset import LM_TRAIN_DATASET


@dataclass(kw_only=True)
class LMTrainConfig:
    model: ModelSection = field(
        default_factory=lambda: ModelSection(
            family="llama",
            arch="llama3_8b",
            compile=True,
            compile_options=CompileOptions(fullgraph=True, dynamic=False),
        )
    )

    dataset: LMTrainDatasetSection = field(
        default_factory=lambda: LMTrainDatasetSection(family=LM_TRAIN_DATASET),
    )

    tokenizer: TokenizerSection = field(
        default_factory=lambda: TokenizerSection(name="llama3")
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    trainer: TrainerSection = field(
        default_factory=lambda: TrainerSection(
            data_parallelism="fsdp", max_grad_norm=1.0
        )
    )

    optimizer: OptimizerSection = field(
        default_factory=lambda: OptimizerSection(
            name=ADAMW_OPTIMIZER,
            config=AdamWConfig(
                lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, impl="fused"
            ),
        ),
    )

    lr_scheduler: LRSchedulerSection | None = field(
        default_factory=lambda: LRSchedulerSection(
            name=COSINE_ANNEALING_LR,
            config=CosineAnnealingLRConfig(
                num_warmup_steps=2000, start_lr=1e-30, final_lr_scale=0.01
            ),
        ),
    )

    regime: RegimeSection = field(
        default_factory=lambda: RegimeSection(
            num_steps=48_000,
            validate_every_n_steps=4000,
            checkpoint_every_n_steps=1000,
            keep_last_n_checkpoints=3,
            keep_checkpoint_every_n_steps=4000,
            publish_metrics_every_n_steps=10,
        )
    )

    common: CommonSection = field(
        default_factory=lambda: CommonSection(
            torch=TorchConfig(
                default_sdpa="flash2", compiled_region_activation_memory_budget=0.9
            )
        )
    )


@dataclass(kw_only=True)
class LMTrainDatasetSection(DatasetSection):
    max_seq_len: int = 8192
    """The maximum sequence length."""

    max_num_tokens: int = 8192 * 2
    """The maximum number of tokens per batch."""

    prefetch: int = 4
    """The number of batches to prefetch in background."""

    sync_ranks: bool = False
