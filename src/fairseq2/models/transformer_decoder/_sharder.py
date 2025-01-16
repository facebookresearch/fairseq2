# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.gang import Gangs
from fairseq2.models.transformer import TransformerEmbeddingFrontend
from fairseq2.models.transformer_decoder._model import TransformerDecoderModel
from fairseq2.nn import (
    ColumnShardedLinear,
    Linear,
    RowShardedLinear,
    ShardedEmbedding,
    StandardEmbedding,
    VocabShardedEmbedding,
)
from fairseq2.nn.transformer import (
    GLUFeedForwardNetwork,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
)

# mypy: disable-error-code="arg-type"


def shard_transformer_decoder_model(
    model: TransformerDecoderModel, gangs: Gangs, shard_embed_dim: bool
) -> None:
    """Shard ``model`` over ``gangs`` for tensor parallelism.

    :param model:
        The model to shard.
    :param gangs:
        The gang used for parallelism.
    :param shard_embed_dim:
        If ``True``, shards :class:`StandardEmbedding` instances over the
        embedding dimension; otherwise, over the vocabulary dimension.
    """
    tp_gang = gangs.tp  # tensor parallel
    if tp_gang.size == 1:
        return

    def shard_embed_frontend(m: TransformerEmbeddingFrontend) -> None:
        if not isinstance(m.embed, StandardEmbedding):
            return

        if shard_embed_dim:
            m.embed = ShardedEmbedding.from_embedding(m.embed, tp_gang)
        else:
            m.embed = VocabShardedEmbedding.from_embedding(m.embed, tp_gang)

    def shard_mha(m: StandardMultiheadAttention) -> None:
        for proj in (m.q_proj, m.k_proj, m.v_proj, m.output_proj):
            if not isinstance(proj, Linear):
                return

        # Scatter.
        m.q_proj = ColumnShardedLinear.from_linear(
            m.q_proj, tp_gang, gather_output=False
        )
        m.k_proj = ColumnShardedLinear.from_linear(
            m.k_proj, tp_gang, gather_output=False
        )
        m.v_proj = ColumnShardedLinear.from_linear(
            m.v_proj, tp_gang, gather_output=False
        )

        # Gather.
        m.output_proj = RowShardedLinear.from_linear(
            m.output_proj, tp_gang, scatter_input=False
        )

        m.num_heads = m.num_heads // tp_gang.size
        m.num_key_value_heads = m.num_key_value_heads // tp_gang.size

    def shard_ffn(m: StandardFeedForwardNetwork) -> None:
        for proj in (m.inner_proj, m.outer_proj):
            if not isinstance(proj, Linear):
                return

        # Scatter.
        m.inner_proj = ColumnShardedLinear.from_linear(
            m.inner_proj, tp_gang, gather_output=False
        )

        # Gather.
        m.output_proj = RowShardedLinear.from_linear(
            m.output_proj, tp_gang, scatter_input=False
        )

    def shard_glu_ffn(m: GLUFeedForwardNetwork) -> None:
        for proj in (m.gate_proj, m.inner_proj, m.output_proj):
            if not isinstance(proj, Linear):
                return

        # Scatter.
        m.gate_proj = ColumnShardedLinear.from_linear(
            m.gate_proj, tp_gang, gather_output=False
        )

        m.inner_proj = ColumnShardedLinear.from_linear(
            m.inner_proj, tp_gang, gather_output=False
        )

        # Gather.
        m.output_proj = RowShardedLinear.from_linear(
            m.output_proj, tp_gang, scatter_input=False
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
        model.final_proj = ColumnShardedLinear.from_linear(model.final_proj, tp_gang)
