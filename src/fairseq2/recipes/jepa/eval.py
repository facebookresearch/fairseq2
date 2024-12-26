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
from fairseq2.models.jepa import load_jepa_model
from fairseq2.models.jepa.classifier import load_jepa_classifier_model
from fairseq2.nn.utils.module import share_parameters
from fairseq2.recipes.utils.asset import AssetReference, retrieve_asset_card
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
    probe_model: AssetReference = ""
    pretrained_model: AssetReference = ""


def evaluate_jepa_attentive_probing(probe: str, pretrain: str, num_classes: int) -> None:
    
    gang = setup_root_gang(log)
    
    config = JepaProbingEvalConfig(
        num_classes=num_classes, probe_model=probe, pretrained_model=pretrain
    )
    
    # Load a pretrained model config to a classifier, then update
    # the attentive pooler and head with the attentive checkpoint
    probe_card = retrieve_asset_card(Path(config.probe_model))
    
    model = load_jepa_classifier_model(probe_card, device=gang.device, dtype=torch.float32)    
    pt_model = load_jepa_model(config.pretrained_model, device=gang.device)
    share_parameters(pt_model.encoder, model.encoder)


if __name__ == "__main__":
    from fire import Fire
    Fire()