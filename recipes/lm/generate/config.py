# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from fairseq2.recipe.config import (
    CommonSection,
    DatasetSection,
    GangSection,
    GeneratorSection,
    ReferenceModelSection,
    SequenceGeneratorSection,
    TokenizerSection,
)

from .dataset import LM_GENERATE_DATASET_FAMILY, LMGenerateDatasetConfig


@dataclass(kw_only=True)
class LMGenerateConfig:
    model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(
            name="llama3_2_1b_instruct", dtype=torch.bfloat16
        )
    )

    dataset: LMGenerateDatasetSection = field(
        default_factory=lambda: LMGenerateDatasetSection(
            family=LM_GENERATE_DATASET_FAMILY,
            config_overrides=LMGenerateDatasetConfig(
                paths=[Path("~/train.jsonl")],
            ),
        )
    )

    tokenizer: TokenizerSection = field(
        default_factory=lambda: TokenizerSection(name="llama3_instruct")
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    generator: GeneratorSection = field(default_factory=lambda: GeneratorSection())

    seq_generator: SequenceGeneratorSection = field(
        default_factory=lambda: SequenceGeneratorSection()
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())


@dataclass(kw_only=True)
class LMGenerateDatasetSection(DatasetSection):
    batch_size: int = 1
    prefetch: int = 4
