# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from fairseq2.config_registry import ConfigRegistry


@dataclass
class HFEvalConfig:
    # Data
    dataset_name: str
    """The HF dataset to evaluate with."""

    # Model
    model_name: str
    """The name of the model to evaluate."""


hf_presets = ConfigRegistry[HFEvalConfig]()
