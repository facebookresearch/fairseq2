# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch

from fairseq2.recipe.config import (
    ADAMW_OPTIMIZER,
    POLYNOMIAL_DECAY_LR,
    AdamWConfig,
    CommonSection,
    CompileOptionsSection,
    GangSection,
    LRSchedulerSection,
    ModelSection,
    OptimizerSection,
    PolynomialDecayLRConfig,
    RegimeSection,
    TrainerSection,
)

from .criterion import Wav2Vec2SslLossSection
from .data import Wav2Vec2SslDatasetSection


@dataclass(kw_only=True)
class Wav2Vec2SslRecipeConfig:
    """
    The default values correspond to the base ls960h training setup as described
    in :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.
    """

    model: ModelSection = field(
        default_factory=lambda: ModelSection(
            family="wav2vec2",
            arch="base",
            compile=False,
            compile_options=CompileOptionsSection(fullgraph=False, dynamic=False),
        )
    )

    dataset: Wav2Vec2SslDatasetSection = field(
        default_factory=lambda: Wav2Vec2SslDatasetSection()
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    trainer: TrainerSection = field(
        default_factory=lambda: TrainerSection(dtype=torch.float16)
    )

    loss: Wav2Vec2SslLossSection = field(
        default_factory=lambda: Wav2Vec2SslLossSection()
    )

    optimizer: OptimizerSection = field(
        default_factory=lambda: OptimizerSection(
            name=ADAMW_OPTIMIZER,
            config=AdamWConfig(
                lr=5e-04, betas=(0.9, 0.98), eps=1e-06, weight_decay=0.01
            ),
        )
    )

    lr_scheduler: LRSchedulerSection = field(
        default_factory=lambda: LRSchedulerSection(
            name=POLYNOMIAL_DECAY_LR,
            config=PolynomialDecayLRConfig(num_warmup_steps=32_000),
        )
    )

    regime: RegimeSection = field(
        default_factory=lambda: RegimeSection(
            num_steps=400_000,
            validate_every_n_steps=5_000,
            checkpoint_every_n_steps=25_000,
            publish_metrics_every_n_steps=200,
        )
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())
