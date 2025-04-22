# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast

from fairseq2.gang import Gangs
from fairseq2.models.transformer import (
    GLUFeedForwardNetwork,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    TransformerEmbeddingFrontend,
)
from fairseq2.nn import (
    ColumnShardedLinear,
    Linear,
    RowShardedLinear,
    ShardedEmbedding,
    StandardEmbedding,
    VocabShardedEmbedding,
)

# isort: split

from fairseq2.models.transformer_lm._model import TransformerLanguageModel


def shard_transformer_lm(
    model: TransformerLanguageModel, gangs: Gangs, *, shard_embed_dim: bool = False
) -> None:
    """
    Shards ``model`` over ``gangs`` for tensor parallelism.

    :param model: The model to shard.
    :param gangs: The gangs used for parallelism.
    :param shard_embed_dim: If ``True``, shards :class:`StandardEmbedding`
        modules over the embedding dimension; otherwise, over the vocabulary
        dimension.
    """
    if gangs.tp.size == 1:
        return

    def shard_embed_frontend(m: TransformerEmbeddingFrontend) -> None:
        if not isinstance(m.embed, StandardEmbedding):
            return

        if shard_embed_dim:
            m.embed = ShardedEmbedding.from_embedding(m.embed, gangs.tp)
        else:
            m.embed = VocabShardedEmbedding.from_embedding(m.embed, gangs.tp)

    def shard_mha(m: StandardMultiheadAttention) -> None:
        for proj in (m.q_proj, m.k_proj, m.v_proj, m.output_proj):
            if not isinstance(proj, Linear):
                return

        # Scatter.
        m.q_proj = ColumnShardedLinear.from_linear(
            cast(Linear, m.q_proj), gangs.tp, gather_output=False
        )
        m.k_proj = ColumnShardedLinear.from_linear(
            cast(Linear, m.k_proj), gangs.tp, gather_output=False
        )
        m.v_proj = ColumnShardedLinear.from_linear(
            cast(Linear, m.v_proj), gangs.tp, gather_output=False
        )

        # Gather.
        m.output_proj = RowShardedLinear.from_linear(
            cast(Linear, m.output_proj), gangs.tp, scatter_input=False
        )

        m.num_heads = m.num_heads // gangs.tp.size
        m.num_key_value_heads = m.num_key_value_heads // gangs.tp.size

    def shard_ffn(m: StandardFeedForwardNetwork) -> None:
        for proj in (m.inner_proj, m.outer_proj):
            if not isinstance(proj, Linear):
                return

        # Scatter.
        m.inner_proj = ColumnShardedLinear.from_linear(
            cast(Linear, m.inner_proj), gangs.tp, gather_output=False
        )

        # Gather.
        m.output_proj = RowShardedLinear.from_linear(
            cast(Linear, m.output_proj), gangs.tp, scatter_input=False
        )

    def shard_glu_ffn(m: GLUFeedForwardNetwork) -> None:
        for proj in (m.gate_proj, m.inner_proj, m.output_proj):
            if not isinstance(proj, Linear):
                return

        # Scatter.
        m.gate_proj = ColumnShardedLinear.from_linear(
            cast(Linear, m.gate_proj), gangs.tp, gather_output=False
        )

        m.inner_proj = ColumnShardedLinear.from_linear(
            cast(Linear, m.inner_proj), gangs.tp, gather_output=False
        )

        # Gather.
        m.output_proj = RowShardedLinear.from_linear(
            cast(Linear, m.output_proj), gangs.tp, scatter_input=False
        )

    for m in model.modules():
        if isinstance(m, TransformerEmbeddingFrontend):
            shard_embed_frontend(m)

            continue

        if isinstance(m, StandardMultiheadAttention):
            shard_mha(m)

            continue

        if isinstance(m, StandardFeedForwardNetwork):
            shard_ffn(m)

            continue

        if isinstance(m, GLUFeedForwardNetwork):
            shard_glu_ffn(m)

            continue

    if isinstance(model.final_proj, Linear):
        model.final_proj = ColumnShardedLinear.from_linear(model.final_proj, gangs.tp)
