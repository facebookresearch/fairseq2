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
    TokenizerSection,
)

from ..data import Wav2Vec2AsrDatasetSection


@dataclass(kw_only=True)
class Wav2Vec2AsrEvalRecipeConfig:
    """wav2vec2 ASR evaluation configuration."""

    # ReferenceModelSection instead of ModelSection because we are
    # loading a checkpoint instead of training the model.
    model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(
            name="wav2vec2_asr_base_10h",
        )
    )

    dataset: Wav2Vec2AsrDatasetSection = field(
        default_factory=lambda: Wav2Vec2AsrDatasetSection(
            # eval specific defaults
            train_split=None,
            valid_split="test_clean",
            batch_shuffle_window=1,
            num_accumulate=1,
            sync_mode=SyncMode.UNTIL_LAST,
        )
    )

    tokenizer: TokenizerSection = field(
        default_factory=lambda: TokenizerSection(name="librispeech_asr")
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    evaluator: EvaluatorSection = field(
        default_factory=lambda: EvaluatorSection(amp=True, amp_dtype=torch.float16)
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())
