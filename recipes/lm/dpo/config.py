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
    ActivationCheckpointingConfig
)

from .dataset import LM_DPO_DATASET, LMDPODatasetConfig, LMDPODataSource


@dataclass(kw_only=True)
class LMDPOConfig:
    model: ModelSection = field(
        default_factory=lambda: ModelSection(name="llama3_1_8b_instruct")
    )
    
    tokenizer: TokenizerSection = field(
        default_factory=lambda: TokenizerSection(name="llama3_instruct")
    )
    
    

    dataset: LMDPODatasetSection = field(
        default_factory=lambda: LMDPODatasetSection(
            family=LM_DPO_DATASET,
            batch_size=16,
            config_overrides=LMDPODatasetConfig(path="hg://facebook/fairseq2-lm-gsm8k"),
        )
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    trainer: TrainerSection = field(
        default_factory=lambda: TrainerSection(
            data_parallelism="fsdp", max_grad_norm=1.0,
            activation_checkpointing=ActivationCheckpointingConfig(mode="layerwise"),
        )
    )

    optimizer: OptimizerSection = field(
        default_factory=lambda: OptimizerSection(
            name=ADAMW_OPTIMIZER,
            config=AdamWConfig(
                lr=5.5e-06, betas=(0.9, 0.95), weight_decay=0.1, impl="fused"
            ),
        )
    )

    lr_scheduler: LRSchedulerSection | None = field(
        default_factory=lambda: LRSchedulerSection(
            name=COSINE_ANNEALING_LR, config=CosineAnnealingLRConfig(final_lr_scale=0.2)
        )
    )

    regime: RegimeSection = field(
        default_factory=lambda: RegimeSection(
            num_steps=5000,
            validate_every_n_steps=100,
            checkpoint_every_n_steps=1000,
            keep_last_n_checkpoints=1,
            publish_metrics_every_n_steps=10,
            export_hugging_face=True,
        )
    )

    # The memory efficient SDPA implementation in PyTorch is numerically not
    # stable when used with padded inputs.
    common: CommonSection = field(
        default_factory=lambda: CommonSection(
            torch=TorchConfig(default_sdpa="torch_math")
        )
    )
    
    dataset: LMDPODatasetSection = field(
        default_factory=lambda: LMDPODatasetSection(family=LM_DPO_DATASET),
    )


@dataclass(kw_only=True)
class LMDPODatasetSection(DatasetSection):
    path: str | None = None

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
