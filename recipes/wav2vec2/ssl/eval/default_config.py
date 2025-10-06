# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch

from fairseq2.datasets import SyncMode
from fairseq2.recipe.config import (
    CommonSection,
    EvaluatorSection,
    GangSection,
    ReferenceModelSection,
)

from ..criterion import Wav2Vec2SslLossSection
from ..data import Wav2Vec2SslDatasetSection


@dataclass(kw_only=True)
class Wav2Vec2SslEvalRecipeConfig:
    """
    Configuration for wav2vec2 SSL evaluation runner.
    """

    # ReferenceModelSection instead of ModelSection because we are
    # loading a checkpoint instead of training the model.
    model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(
            name="wav2vec2_base",
        )
    )

    dataset: Wav2Vec2SslDatasetSection = field(
        default_factory=lambda: Wav2Vec2SslDatasetSection(
            # eval specific defaults
            train_split=None,
            valid_split="valid",
            batch_shuffle_window=1,
            num_accumulate=1,
            sync_mode=SyncMode.UNTIL_LAST,
        )
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    evaluator: EvaluatorSection = field(
        default_factory=lambda: EvaluatorSection(amp=True, amp_dtype=torch.float16)
    )

    loss: Wav2Vec2SslLossSection = field(
        default_factory=lambda: Wav2Vec2SslLossSection()
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())
