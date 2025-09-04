# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import Module

from fairseq2.models.utils.ac import apply_layerwise_ac
from fairseq2.models.wav2vec2.asr.model import Wav2Vec2AsrModel


def apply_ac_to_wav2vec2_asr(model: Wav2Vec2AsrModel, every_nth_layer: int) -> Module:
    apply_layerwise_ac(model.encoder.layers, every_nth_layer)

    return model
