# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import Module

from fairseq2.models.gemma4.decoder import Gemma4Decoder
from fairseq2.models.gemma4.model import Gemma4Model
from fairseq2.models.utils.ac import apply_layerwise_ac


def apply_ac_to_gemma4(model: Gemma4Model, every_nth_layer: int) -> Module:
    decoder = model.decoder

    if not isinstance(decoder, Gemma4Decoder):
        raise TypeError(f"Expected Gemma4Decoder, got {type(decoder)}")

    apply_layerwise_ac(decoder.layers, every_nth_layer)

    return model
