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

from .dataset import LM_SFT_DATASET, LMSFTDatasetConfig, LMSFTDataSource


@dataclass(kw_only=True)
class LMSFTConfig:
    model: ModelSection = field(
        default_factory=lambda: ModelSection(name="llama3_2_1b")
    )

    dataset: LMSFTDatasetSection = field(
        default_factory=lambda: LMSFTDatasetSection(
            family=LM_SFT_DATASET,
            config_overrides=LMSFTDatasetConfig(
                sources={
                    "train": [
                        LMSFTDataSource(
                            path="hg://facebook/fairseq2-lm-gsm8k",
                            split="sft_train",
                            weight=1.0,
                        ),
                    ],
                },
            ),
        )
    )

    tokenizer: TokenizerSection = field(
        default_factory=lambda: TokenizerSection(name="llama3_2_1b_instruct")
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
            config=CosineAnnealingLRConfig(final_lr_scale=0.2),
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
            export_hugging_face=True,
            save_model_only="all_but_last",
        )
    )

    common: CommonSection = field(
        default_factory=lambda: CommonSection(
            torch=TorchConfig(default_sdpa="torch_math")
        )
    )


@dataclass(kw_only=True)
class LMSFTDatasetSection(DatasetSection):
    train_split: str = "sft_train"

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
    """The maximum sequence length. NOTE: longer sequences are dropped from the training."""

    max_num_tokens: int = 8192 * 2
    """The maximum number of tokens per batch. NOTE: this is excluding padding tokens!"""

    batch_size: int | None = None
    """
    If not ``None``, ignores ``max_num_tokens`` and each batch will have
    ``batch_size`` examples.
    """

    max_num_valid_tokens: int | None = None
    """The maximum number of tokens per validation batch."""

    example_shuffle_window: int = 10_000
    """The size of the sliding window for shuffling examples."""

    batch_shuffle_window: int = 0
    """The size of the sliding window for shuffling batches."""

    prefetch: int = 4
    """The number of batches to prefetch in background."""
