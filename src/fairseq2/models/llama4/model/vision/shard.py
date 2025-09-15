# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

from fairseq2.gang import Gang
from fairseq2.models.llama4.model.vision.attention import Attention
from fairseq2.models.llama4.model.vision.embedding import VisionEmbeddings
from fairseq2.models.llama4.model.vision.encoder import Conv2dPatch
from fairseq2.models.llama4.model.vision.ffn import _FeedForward
from fairseq2.nn import ColumnShardedLinear, Linear, RowShardedLinear

# mypy: disable-error-code="arg-type"


def shard_vision_embedding(model: VisionEmbeddings, tp_gang: Gang) -> None:
    def shard_ffn(m: _FeedForward) -> None:
        for proj in (m.c_fc, m.c_proj):
            if not isinstance(proj, Linear):
                return

        m.c_fc = ColumnShardedLinear.from_linear(m.c_fc, tp_gang, gather_output=False)
        m.c_proj = RowShardedLinear.from_linear(m.c_proj, tp_gang, scatter_input=False)

    def shard_mha(m: Attention) -> None:
        for proj in (m.wq, m.wk, m.wv, m.wo):
            if not isinstance(proj, Linear):
                return

        # Scatter.
        m.wq = ColumnShardedLinear.from_linear(m.wq, tp_gang, gather_output=False)
        m.wk = ColumnShardedLinear.from_linear(m.wk, tp_gang, gather_output=False)
        m.wv = ColumnShardedLinear.from_linear(m.wv, tp_gang, gather_output=False)

        # Gather.
        m.wo = RowShardedLinear.from_linear(m.wo, tp_gang, scatter_input=False)

        m.n_local_heads = m.n_heads // tp_gang.size
        m.n_local_kv_heads = m.n_kv_heads // tp_gang.size

        # Shard the kv-cache too
        m.init_kv_cache()

    def shard_conv(m: Conv2dPatch) -> None:
        m._linear = ColumnShardedLinear.from_linear(m._linear, tp_gang)

    for m in model.modules():
        if isinstance(m, Conv2dPatch):
            shard_conv(m)

            continue

        if isinstance(m, Attention):
            shard_mha(m)

            continue

        if isinstance(m, _FeedForward):
            shard_ffn(m)

            continue
