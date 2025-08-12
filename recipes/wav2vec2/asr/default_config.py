# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch

from fairseq2.recipe.config import (
    ADAMW_OPTIMIZER,
    TRI_STAGE_LR,
    AdamWConfig,
    CommonSection,
    CompileOptionsSection,
    GangSection,
    GradAccumulationSection,
    LRSchedulerSection,
    ModelSection,
    OptimizerSection,
    ReferenceModelSection,
    RegimeSection,
    TokenizerSection,
    TrainerSection,
    TriStageLRConfig,
)

from .data import Wav2Vec2AsrDatasetSection


@dataclass(kw_only=True)
class Wav2Vec2AsrTrainerSection(TrainerSection):
    """
    ASR-specific trainer configuration with encoder freezing.
    Note the inheritance from TrainerSection which can be used for recipe customization.
    """

    freeze_encoder_for_n_steps: int = 10_000
    """The encoder will be frozen for this number of steps."""


@dataclass(kw_only=True)
class Wav2Vec2AsrConfig:
    """
    wav2vec2 ASR training configuration.
    The default values correspond to the base_10h training setup.
    """

    # Model configuration
    model: ModelSection = field(
        default_factory=lambda: ModelSection(
            family="wav2vec2_asr",
            arch="base_10h",
            compile=False,
            compile_options=CompileOptionsSection(fullgraph=False, dynamic=False),
        )
    )

    # Pretrained model (needed if we start training from that checkpoint to share the encoder)
    pretrained_model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(
            name="wav2vec2_base",
            compile=False,
            compile_options=CompileOptionsSection(fullgraph=False, dynamic=False),
        )
    )

    dataset: Wav2Vec2AsrDatasetSection = field(
        default_factory=lambda: Wav2Vec2AsrDatasetSection(
            batch_shuffle_window=1,
            example_shuffle_window=1,  # TODO: set deterministically for debugging
        )
    )

    tokenizer: TokenizerSection = field(
        default_factory=lambda: TokenizerSection(name="librispeech_asr")
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    trainer: Wav2Vec2AsrTrainerSection = field(
        default_factory=lambda: Wav2Vec2AsrTrainerSection(
            dtype=torch.float16,
            grad_accumulation=GradAccumulationSection(num_batches=4),
        )
    )

    optimizer: OptimizerSection = field(
        default_factory=lambda: OptimizerSection(
            name=ADAMW_OPTIMIZER,
            config=AdamWConfig(
                lr=5e-05,
                betas=(0.9, 0.98),
                eps=1e-08,
                weight_decay=0.00,
            ),
        )
    )

    lr_scheduler: LRSchedulerSection = field(
        default_factory=lambda: LRSchedulerSection(
            name=TRI_STAGE_LR,
            config=TriStageLRConfig(
                stage_ratio=(0.1, 0.4, 0.5),
                start_lr_scale=0.01,
                final_lr_scale=0.05,
            ),
        )
    )

    regime: RegimeSection = field(
        default_factory=lambda: RegimeSection(
            num_steps=20_000,
            score_metric="wer",  # defined in wer_calculator.py::WerCalculator::_wer_key
            validate_after_n_steps=10_000,
            validate_every_n_steps=1_000,
            publish_metrics_every_n_steps=200,
            checkpoint_every_n_steps=200_000,  # default 5_000, but checkpointing is stil broken TODO: cirquit
        )
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())
