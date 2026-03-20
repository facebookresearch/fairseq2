# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import Module

from fairseq2.error import NotSupportedError
from fairseq2.models.gemma3n.decoder import Gemma3nDecoder
from fairseq2.models.gemma3n.model import Gemma3nModel
from fairseq2.models.utils.fsdp import apply_layerwise_fsdp
from fairseq2.nn.fsdp import FSDPWrapper


def apply_fsdp_to_gemma3n(
    model: Gemma3nModel, granularity: str, wrapper: FSDPWrapper
) -> Module:
    decoder = model.decoder

    if not isinstance(decoder, Gemma3nDecoder):
        raise TypeError(f"Expected Gemma3nDecoder, got {type(decoder)}")

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
