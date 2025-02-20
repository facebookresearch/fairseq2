# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.gang import Gangs
from fairseq2.models.llama._config import LLaMAConfig
from fairseq2.models.transformer_decoder import (
    TransformerDecoderModel,
    shard_transformer_decoder_model,
)


def shard_llama_model(
    model: TransformerDecoderModel, config: LLaMAConfig, gangs: Gangs
) -> None:
    shard_embed_dim = config.max_seq_len < 8192  # LLaMA 1 or 2

    shard_transformer_decoder_model(model, gangs, shard_embed_dim)
