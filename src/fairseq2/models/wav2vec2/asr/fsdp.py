# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import Module

from fairseq2.error import NotSupportedError
from fairseq2.models.utils.fsdp import apply_layerwise_fsdp
from fairseq2.models.wav2vec2.asr.model import Wav2Vec2AsrModel
from fairseq2.nn.fsdp import FSDPWrapper


def apply_fsdp_to_wav2vec2_asr(
    model: Wav2Vec2AsrModel, granularity: str, wrapper: FSDPWrapper
) -> Module:
    if granularity == "layer":
        apply_layerwise_fsdp(model.encoder.layers, wrapper)

        return model

    if granularity == "stack":
        wrapped_encoder = wrapper(model.encoder)

        model.register_module("encoder", wrapped_encoder)

        return model

    raise NotSupportedError(
        f"`granularity` must be a supported granularity, but is {granularity} instead."
    )
