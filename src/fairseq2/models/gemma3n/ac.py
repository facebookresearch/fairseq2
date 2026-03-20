# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import Module

from fairseq2.models.gemma3n.decoder import Gemma3nDecoder
from fairseq2.models.gemma3n.model import Gemma3nModel
from fairseq2.models.utils.ac import apply_layerwise_ac


def apply_ac_to_gemma3n(model: Gemma3nModel, every_nth_layer: int) -> Module:
    decoder = model.decoder

    if not isinstance(decoder, Gemma3nDecoder):
        raise TypeError(f"Expected Gemma3nDecoder, got {type(decoder)}")

    apply_layerwise_ac(decoder.layers, every_nth_layer)

    return model
