# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch import Tensor
from typing_extensions import override

from fairseq2.data import VocabularyInfo
from fairseq2.gang import Gang
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.models.transformer.frontend import (
    TransformerEmbeddingFrontend,
    TransformerFrontend,
)
from fairseq2.nn import (
    ColumnShardedLinear,
    Linear,
    Projection,
    RowShardedLinear,
    ShardedEmbedding,
    StandardEmbedding,
    VocabShardedEmbedding,
)
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer import (
    GLUFeedForwardNetwork,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    TransformerDecoder,
)


@final
class TransformerDecoderModel(DecoderModel):
    """Represents a Transformer-based decoder model."""

    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    final_proj: Projection

    def __init__(
        self,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        max_seq_len: int,
        vocab_info: VocabularyInfo,
    ) -> None:
        """
        :param decoder_frontend:
            The decoder frontend.
        :param decoder:
            The decoder.
        :param final_proj:
            The projection to apply to decoder outputs.
        :param max_seq_len:
            The maximum length of sequences produced by the model.
        :param vocab_info:
            The vocabulary information of sequences produced by the model.
        """
        super().__init__(decoder.model_dim, max_seq_len, vocab_info)

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder

        self.final_proj = final_proj

    @override
    def decode(
        self,
        seqs: Tensor,
        padding_mask: PaddingMask | None,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> tuple[Tensor, PaddingMask]:
        seqs, padding_mask = self.decoder_frontend(
            seqs, padding_mask, state_bag=state_bag
        )

        decoder_output, decoder_padding_mask = self.decoder(
            seqs, padding_mask, state_bag=state_bag
        )

        return decoder_output, decoder_padding_mask

    @override
    def project(
        self, decoder_output: Tensor, decoder_padding_mask: PaddingMask | None
    ) -> SequenceModelOutput:
        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.vocab_info.pad_idx)


# mypy: disable-error-code="arg-type"


def shard_transformer_decoder_model(
    model: TransformerDecoderModel, gang: Gang, shard_embed_dim: bool = False
) -> None:
    """Shard ``model`` over ``gang`` for tensor parallelism.

    :param model:
        The model to shard.
    :param gang:
        The gang used for tensor parallelism.
    :param shard_embed_dim:
        If ``True``, shards :class:`StandardEmbedding` instances over the
        embedding dimension; otherwise, over the vocabulary dimension.
    """
    if gang.size == 1:
        return

    def shard_embed_frontend(m: TransformerEmbeddingFrontend) -> None:
        if not isinstance(m.embed, StandardEmbedding):
            return

        if shard_embed_dim:
            m.embed = ShardedEmbedding.from_embedding(m.embed, gang)
        else:
            m.embed = VocabShardedEmbedding.from_embedding(m.embed, gang)

    def shard_mha(m: StandardMultiheadAttention) -> None:
        for proj in (m.q_proj, m.k_proj, m.v_proj, m.output_proj):
            if not isinstance(proj, Linear):
                return

        # Scatter.
        m.q_proj = ColumnShardedLinear.from_linear(m.q_proj, gang, gather_output=False)
        m.k_proj = ColumnShardedLinear.from_linear(m.k_proj, gang, gather_output=False)
        m.v_proj = ColumnShardedLinear.from_linear(m.v_proj, gang, gather_output=False)

        # Gather.
        m.output_proj = RowShardedLinear.from_linear(
            m.output_proj, gang, scatter_input=False
        )

        m.num_heads = m.num_heads // gang.size
        m.num_key_value_heads = m.num_key_value_heads // gang.size

    def shard_ffn(m: StandardFeedForwardNetwork) -> None:
        for proj in (m.inner_proj, m.outer_proj):
            if not isinstance(proj, Linear):
                return

        # Scatter.
        m.inner_proj = ColumnShardedLinear.from_linear(
            m.inner_proj, gang, gather_output=False
        )

        # Gather.
        m.output_proj = RowShardedLinear.from_linear(
            m.output_proj, gang, scatter_input=False
        )

    def shard_glu_ffn(m: GLUFeedForwardNetwork) -> None:
        for proj in (m.gate_proj, m.inner_proj, m.output_proj):
            if not isinstance(proj, Linear):
                return

        # Scatter.
        m.gate_proj = ColumnShardedLinear.from_linear(
            m.gate_proj, gang, gather_output=False
        )

        m.inner_proj = ColumnShardedLinear.from_linear(
            m.inner_proj, gang, gather_output=False
        )

        # Gather.
        m.output_proj = RowShardedLinear.from_linear(
            m.output_proj, gang, scatter_input=False
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
        model.final_proj = ColumnShardedLinear.from_linear(model.final_proj, gang)
