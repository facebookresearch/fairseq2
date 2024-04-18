# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, final

from torch import Tensor

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
    Embedding,
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
    StandardMultiheadAttention,
    TransformerDecoder,
)
from fairseq2.typing import override


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
        padding_mask: Optional[PaddingMask],
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, PaddingMask]:
        seqs, padding_mask = self.decoder_frontend(
            seqs, padding_mask, state_bag=state_bag
        )

        decoder_output, decoder_padding_mask = self.decoder(
            seqs, padding_mask, state_bag=state_bag
        )

        return decoder_output, decoder_padding_mask

    @override
    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[PaddingMask]
    ) -> SequenceModelOutput:
        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.vocab_info.pad_idx)


def shard_transformer_decoder_model(
    model: TransformerDecoderModel, gang: Gang, shard_embed_dim: bool = False
) -> None:
    """Shard ``model`` over ``gang``.

    :param model:
        The model to shard.
    :param gang:
        The gang used for tensor parallelism.
    :param shard_embed_dim:
        If ``True``, shards :class:`StandardEmbedding` instances across
        embedding dimension instead of vocabulary dimension.
    """
    if gang.size == 1:
        return

    def shard_embedding(embed: Embedding) -> Embedding:
        if isinstance(embed, StandardEmbedding):
            if shard_embed_dim:
                return ShardedEmbedding.from_embedding(embed, gang)

            return VocabShardedEmbedding.from_embedding(embed, gang)

        return embed

    def row_shard(proj: Projection) -> Projection:
        if isinstance(proj, Linear):
            return RowShardedLinear.from_linear(proj, gang)

        return proj

    def column_shard(proj: Projection) -> Projection:
        if isinstance(proj, Linear):
            return ColumnShardedLinear.from_linear(proj, gang)

        return proj

    for m in model.modules():
        if isinstance(m, TransformerEmbeddingFrontend):
            m.embed = shard_embedding(m.embed)

            continue

        if isinstance(m, StandardMultiheadAttention):
            m.q_proj = column_shard(m.q_proj)
            m.k_proj = column_shard(m.k_proj)
            m.v_proj = column_shard(m.v_proj)

            m.output_proj = row_shard(m.output_proj)

            continue

        if isinstance(m, GLUFeedForwardNetwork):
            m.gate_proj = column_shard(m.gate_proj)

            m.inner_proj = column_shard(m.inner_proj)

            m.output_proj = row_shard(m.output_proj)

            continue

    model.final_proj = column_shard(model.final_proj)
