# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor parallelism shard specifications for NemotronH."""

from __future__ import annotations

from fairseq2.models.transformer_lm import TransformerLM
from fairseq2.nn import ShardSpec


def get_nemotron_h_shard_specs(model: TransformerLM) -> list[ShardSpec]:
    """Get tensor parallelism shard specifications for NemotronH.

    TODO: Implement proper sharding for MoE experts and Mamba2.
    """
    return []
