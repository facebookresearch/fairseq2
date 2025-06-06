# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.gang import Gangs
from fairseq2.models.transformer_lm import get_transformer_lm_shard_specs
from fairseq2.models.utils.sharder import ModuleShardSpec

# isort: split

from fairseq2.models.qwen._config import QwenConfig


def get_qwen_shard_specs(config: QwenConfig, gangs: Gangs) -> dict[str, ModuleShardSpec]:
    return get_transformer_lm_shard_specs(gangs)
