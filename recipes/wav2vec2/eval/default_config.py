# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from fairseq2.recipe.config import (
    ADAMW_OPTIMIZER,
    POLYNOMIAL_DECAY_LR,
    CommonSection,
    CompileOptionsSection,
    DatasetSection,
    GangSection,
    LRSchedulerSection,
    ModelSection,
    OptimizerSection,
    PolynomialDecayLRConfig,
    TrainerSection,
)


@dataclass(kw_only=True)
class Wav2Vec2EvalConfig:
    model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(name="wav2vec2_base")
    )

    dataset: Wav2Vec2EvalDatasetSection = field(
        default_factory=lambda: Wav2Vec2EvalDatasetSection()
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    evaluator: EvaluatorSection = field(
        default_factory=lambda: EvaluatorSection(dtype=torch.float16)
    )

    loss: Wav2Vec2LossSection = field(default_factory=lambda: Wav2Vec2LossSection())

    common: CommonSection = field(default_factory=lambda: CommonSection())
