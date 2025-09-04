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
from typing import Any, Final, cast, final
from fairseq2.recipe.config import (
    ADAMW_OPTIMIZER,
    COSINE_ANNEALING_LR,
    AdamWConfig,
    CommonSection,
    CompileOptionsSection,
    CosineAnnealingLRConfig,
    DatasetSection,
    FSDPSection,
    GangSection,
    LRSchedulerSection,
    ModelSection,
    OptimizerSection,
    RegimeSection,
    TokenizerSection,
    TorchSection,
    TrainerSection,
)

from .dataset import LM_SFT_DATASET, LMSFTDatasetConfig

GENERIC_INSTRUCTION_DATASET_FAMILY: Final = "generic_instruction"

@dataclass(kw_only=True)
class InstructionFinetuneDatasetSection(DatasetSection):
    # name: str = "foo"  # TODO: change!

    # family: str = GENERIC_INSTRUCTION_DATASET_FAMILY

    path: Path | None = None

    train_split: str = "train"  # DEFAULT

    valid_split: str | None = None

    source_encode_mode: str = "prompt"
    """The encode mode for the prompt, determines what special tokens to add."""

    target_encode_mode: str = "prompt_response"
    """The encode mode for the target, determines what special tokens to add."""

    chat_mode: bool = False
    """If True, dataset jsonl must have 'chat' field with openai-like messages List[Dict] entries"""

    min_seq_len: int = 1
    """The minimum sequence length."""

    max_seq_len: int = 8192
    """The maximum sequence length."""

    max_num_tokens: int = 8192 * 2
    """The maximum number of tokens per batch."""

    batch_size: int | None = None
    """
    If not ``None``, ignores ``max_num_tokens`` and each batch will have
    ``batch_size`` examples.
    """

    max_num_valid_tokens: int | None = None
    """The maximum number of tokens per validation batch."""

    example_shuffle_window: int = 10_000
    """The size of the sliding window for shuffling examples."""

    batch_shuffle_window: int = 1000
    """The size of the sliding window for shuffling batches."""

    prefetch: int = 4
    """The number of batches to prefetch in background."""

    extras: dict[str, object] = field(default_factory=dict)
    """The dataset-specific extra options."""

@dataclass(kw_only=True)
class LMSFTConfig:
    model: ModelSection = field(
        default_factory=lambda: ModelSection(
            family="llama",
            arch="llama3_2_1b",
            compile=False,
            compile_options=CompileOptionsSection(fullgraph=True, dynamic=False),
        )
    )


    tokenizer: TokenizerSection = field(
        default_factory=lambda: TokenizerSection(name="llama3_2_1b")
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    trainer: TrainerSection = field(
        default_factory=lambda: TrainerSection(
            dtype=torch.bfloat16,
            data_parallelism="fsdp",
            fsdp=FSDPSection(version="v2", fp32_reduce=True),
            max_grad_norm=1.0,
            gc_every_n_steps=1000,
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
            torch=TorchSection(
                default_sdpa="flash2", compiled_region_activation_memory_budget=0.9
            )
        )
    )

    dataset: InstructionFinetuneDatasetSection = field(
        default_factory=lambda: InstructionFinetuneDatasetSection(
            family=LM_SFT_DATASET,
            config_overrides=LMSFTDatasetConfig(
                path=Path(
                    "/checkpoint/ram/jacklanchantin/data/alpaca/"
                )
            ),
        )
    )





    # dataset: LMSFTDatasetSection = field(
    #     default_factory=lambda: LMSFTDatasetSection(
    #         family=LM_SFT_DATASET,
    #         config_overrides=LMSFTDatasetConfig(
    #             path=Path(
    #                 "/checkpoint/ram/jacklanchantin/data/alpaca/"
    #             )
    #         ),
    #     )
    # )

    # dataset: InstructionFinetuneDatasetSection = field(
    #     default_factory=lambda: InstructionFinetuneDatasetSection()
    # )


# @dataclass(kw_only=True)
# class LMSFTDatasetSection(DatasetSection):
#     max_seq_len: int = 8192
#     """The maximum sequence length."""

#     max_num_tokens: int = 8192 * 2
#     """The maximum number of tokens per batch."""

#     prefetch: int = 4
#     """The number of batches to prefetch in background."""

#     sync_ranks: bool = False
