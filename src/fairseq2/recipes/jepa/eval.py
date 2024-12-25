# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from fairseq2.logging import get_log_writer
from fairseq2.models.jepa.classifier import load_jepa_classifier_model
from fairseq2.recipes.utils.asset import AssetReference
from fairseq2.recipes.utils.setup import setup_root_gang

log = get_log_writer(__name__)


@dataclass(kw_only=True)
class JepaProbingEvalConfig:
    """
    Holds the configuration for the evaluation of a Jepa model using
    attentive probing on certain task (Currently only classification).
    """

    # Data
    dataset: AssetReference = "k400_234"
    num_classes: int = 400

    # Model
    model_card: AssetReference = ""


def evaluate_jepa_attentive_probing(
    config: JepaProbingEvalConfig,
    output_dir: Path,
) -> None:
    
    gang = setup_root_gang(log)
    
    # Load a pretrained model config to a classifier, then update
    # the attentive pooler and head with the attentive checkpoint
    model = load_jepa_classifier_model(config.model_card, device=gang.device, dtype=torch.float32)


if __name__ == "__main__":
    from fire import Fire
    Fire()