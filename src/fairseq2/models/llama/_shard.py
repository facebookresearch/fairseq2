# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.transformer_lm import get_transformer_lm_shard_specs
from fairseq2.models.utils.sharder import ShardSpec

# isort: split

from fairseq2.models.llama._config import LLaMAConfig


def get_llama_shard_specs(config: LLaMAConfig) -> dict[str, ShardSpec]:
    return get_transformer_lm_shard_specs(shard_embed_dim=config.shard_embed_dim)
