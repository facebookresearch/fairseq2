# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.gang import Gangs
from fairseq2.models.transformer_lm import (
    TransformerLM,
    shard_transformer_lm,
)

# isort: split

from fairseq2.models.llama._config import LLaMAConfig


def shard_llama_model(model: TransformerLM, config: LLaMAConfig, gangs: Gangs) -> None:
    shard_transformer_lm(model, gangs, shard_embed_dim=config.shard_embed_dim)
