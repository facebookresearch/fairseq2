# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import Module

from fairseq2.error import NotSupportedError
from fairseq2.models.transformer.model import TransformerModel
from fairseq2.models.utils.fsdp import apply_layerwise_fsdp
from fairseq2.nn.fsdp import FSDPWrapper


def apply_fsdp_to_transformer(
    model: TransformerModel, granularity: str, wrapper: FSDPWrapper
) -> Module:
    if granularity == "layer":
        apply_layerwise_fsdp(model.encoder.layers, wrapper)
        apply_layerwise_fsdp(model.decoder.layers, wrapper)

        return model

    if granularity == "stack":
        wrapped_encoder = wrapper(model.encoder)
        wrapped_decoder = wrapper(model.decoder)

        model.register_module("encoder", wrapped_encoder)
        model.register_module("decoder", wrapped_decoder)

        return model

    raise NotSupportedError(
        f"`granularity` must be a supported granularity, but is {granularity} instead."
    )
