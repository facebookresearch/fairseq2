# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import torch

from fairseq2.models.transformer_lm._decoder import StandardTransformerLMDecoder
from fairseq2.models.transformer_lm._model import TransformerLM


def compile_transformer_lm(model: TransformerLM, *args: Any, **kwargs: Any) -> None:
    for layer in model.decoder.layers:
        layer.compile(*args, **kwargs)

    if isinstance(model.decoder, StandardTransformerLMDecoder):
        if model.decoder.layer_norm is not None:
            model.decoder.layer_norm.compile(*args, **kwargs)

    model.compute_fused_loss = torch.compile(model.compute_fused_loss, *args, **kwargs)  # type: ignore[method-assign]
