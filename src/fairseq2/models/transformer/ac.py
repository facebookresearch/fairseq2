# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import Module

from fairseq2.models.transformer.model import TransformerModel
from fairseq2.models.utils.ac import apply_layerwise_ac


def apply_ac_to_transformer(model: TransformerModel, every_nth_layer: int) -> Module:
    apply_layerwise_ac(model.encoder.layers, every_nth_layer)
    apply_layerwise_ac(model.decoder.layers, every_nth_layer)

    return model
