# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from fairseq2.data_type import DataType
from fairseq2.recipe.config import CommonSection, DatasetSection, GangSection

from .dataset import MULTIMODAL_GENERATE_DATASET_FAMILY, MultimodalGenerateDatasetConfig


@dataclass(kw_only=True)
class HGModelSection:
    """HuggingFace model configuration for multimodal inference."""

    hf_name: str = "google/gemma-3-4b-it"
    dtype: DataType = torch.bfloat16
    trust_remote_code: bool = True


@dataclass(kw_only=True)
class HFGenerationSection:
    """HuggingFace .generate() kwargs."""

    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass(kw_only=True)
class VideoSection:
    """Video frame sampling configuration."""

    num_frames: int = 8


@dataclass(kw_only=True)
class MultimodalDatasetSection(DatasetSection):
    batch_size: int = 1
    prefetch: int = 2


@dataclass(kw_only=True)
class MultimodalGenerateConfig:
    model: HGModelSection = field(default_factory=lambda: HGModelSection())

    dataset: MultimodalDatasetSection = field(
        default_factory=lambda: MultimodalDatasetSection(
            family=MULTIMODAL_GENERATE_DATASET_FAMILY,
            config_overrides=MultimodalGenerateDatasetConfig(
                paths=[Path("~/data.jsonl")],
            ),
        )
    )

    generation: HFGenerationSection = field(
        default_factory=lambda: HFGenerationSection()
    )

    video: VideoSection = field(default_factory=lambda: VideoSection())

    gang: GangSection = field(default_factory=lambda: GangSection())

    common: CommonSection = field(default_factory=lambda: CommonSection())
