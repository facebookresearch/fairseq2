# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.qwen.config import QwenConfig
from fairseq2.sharder import ShardSpec
from fairseq2.utils.warn import _warn_deprecated


def get_qwen_shard_specs(config: QwenConfig) -> dict[str, ShardSpec]:
    _warn_deprecated(
        "`get_qwen_shard_specs` is deprecated and will be removed in fairseq2 v0.12. See src/fairseq2/sharder.py for details."
    )

    return {
        # fmt: off
        r".*\.embed$":                 ShardSpec(dim=0),
        r".*\.self_attn.q_proj$":      ShardSpec(dim=0, region_boundary=True),
        r".*\.self_attn.k_proj$":      ShardSpec(dim=0, region_boundary=True),
        r".*\.self_attn.v_proj$":      ShardSpec(dim=0, region_boundary=True),
        r".*\.self_attn.output_proj$": ShardSpec(dim=1, region_boundary=True),
        r".*\.ffn.inner_proj$":        ShardSpec(dim=0, region_boundary=True),
        r".*\.ffn.gate_proj$":         ShardSpec(dim=0, region_boundary=True),
        r".*\.ffn.output_proj$":       ShardSpec(dim=1, region_boundary=True),
        r"^final_proj$":               ShardSpec(dim=0),
        # fmt: on
    }
