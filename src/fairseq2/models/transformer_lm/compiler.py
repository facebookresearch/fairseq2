# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

from fairseq2.models.transformer_lm.model import TransformerLM


def compile_transformer_lm(model: TransformerLM, *args: Any, **kwargs: Any) -> None:
    model.decoder.compile_layerwise(*args, **kwargs)

    model.compile_loss(*args, **kwargs)
