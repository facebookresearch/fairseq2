# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.gang import Gangs
from fairseq2.models.qwen._config import QwenConfig
from fairseq2.models.transformer_decoder import (
    TransformerDecoderModel,
    shard_transformer_decoder_model,
)


def shard_qwen_model(
    model: TransformerDecoderModel, config: QwenConfig, gangs: Gangs
) -> None:

    shard_transformer_decoder_model(model, gangs, shard_embed_dim=False)