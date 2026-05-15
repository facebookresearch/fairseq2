# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import Module

from fairseq2.error import NotSupportedError
from fairseq2.models.gemma4.decoder import Gemma4Decoder
from fairseq2.models.gemma4.model import Gemma4Model
from fairseq2.models.utils.fsdp import apply_layerwise_fsdp
from fairseq2.nn.fsdp import FSDPWrapper


def apply_fsdp_to_gemma4(
    model: Gemma4Model, granularity: str, wrapper: FSDPWrapper
) -> Module:
    decoder = model.decoder

    if not isinstance(decoder, Gemma4Decoder):
        raise TypeError(f"Expected Gemma4Decoder, got {type(decoder)}")

    if granularity == "layer":
        apply_layerwise_fsdp(decoder.layers, wrapper)

        return model

    if granularity == "stack":
        wrapped_decoder = wrapper(decoder)

        model.register_module("decoder", wrapped_decoder)

        return model

    raise NotSupportedError(
        f"`granularity` must be a supported granularity, but is {granularity} instead."
    )
