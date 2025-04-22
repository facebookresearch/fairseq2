# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

from fairseq2.models.transformer_lm._model import TransformerLanguageModel


def compile_transformer_lm(model: TransformerLanguageModel, **kwargs: Any) -> None:
    for layer in model.decoder.layers:
        layer.compile(**kwargs)

    model.compile_loss_function(**kwargs)
