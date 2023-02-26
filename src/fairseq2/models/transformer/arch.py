# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple, final

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import final as finaloverride
from torch import Tensor
from torch.nn import LayerNorm, Module

from fairseq2.nn.embedding import Embedding
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.positional_embedding import PositionalEmbedding
from fairseq2.nn.projection import Projection, ResettableProjection
from fairseq2.nn.transformer.decoder import TransformerDecoder
from fairseq2.nn.transformer.encoder import TransformerEncoder


class TransformerTokenFrontend(Module):
    """Represents a Transformer front-end as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    embed: Embedding
    scale: float
    pos_embed: Optional[PositionalEmbedding]
    embed_norm: Optional[LayerNorm]
    dropout_p: float

    def __init__(
        self,
        embed: Embedding,
        pos_embed: Optional[PositionalEmbedding],
        no_scale: bool = False,
        norm: bool = False,
        dropout_p: float = 0.1,
        norm_eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param embed:
            The token embedding.
        :param pos_embed:
            The positional embedding.
        :param no_scale:
            If ``True``, embeddings won't be scaled by the square root of the
            embedding size.
        :param norm:
            If ``True``, applies Layer Normalization to embeddings.
        :param dropout_p:
            The dropout probability on embeddings.
        :param norm_eps:
            The epsilon value to add to the denominator of the
            :class:`~torch.nn.LayerNorm` modules for numerical stability.
        """
        super().__init__()

        embedding_dim = embed.embedding_dim

        self.embed = embed

        self.scale = 1.0 if no_scale else math.sqrt(embedding_dim)

        if pos_embed is not None:
            if pos_embed.embedding_dim != embedding_dim:
                raise ValueError(
                    f"`embedding_dim` of `pos_embed` ({pos_embed.embedding_dim}) does not match `embedding_dim` of `embed` ({embedding_dim})."
                )

            self.pos_embed = pos_embed
        else:
            self.register_module("pos_embed", None)

        if norm:
            self.norm = LayerNorm(embedding_dim, norm_eps, device=device, dtype=dtype)
        else:
            self.register_module("norm", None)

        self.dropout_p = dropout_p

    def forward(
        self, token_indices: Tensor, state_bag: Optional[IncrementalStateBag] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param token_indices:
            The token indices to process. *Shape:* :math:`(N,S)`, or :math:`(S)`
            when unbatched, where :math:`N` is the batch size and :math:`S` is
            the sequence length.
        :param state_bag:
            The state bag to use during an incremental evaluation.

        :returns:
            - The token embeddings to pass to the encoder or decoder. *Shape:*
              :math:`(N,S,M)`, or :math:`(S,M)` when unbatched, where :math:`N`
              is the batch size, :math:`S` is the sequence length, and :math:`M`
              is the model size.
            - The boolean padding mask indicating which key positions to ignore
              for the purpose of self attention. *Shape:* :math:`(N,S)`, or
              :math:`(S)` when unbatched, where :math:`N` is the batch size and
              :math:`S` is the sequence length.

        .. note::
            For a boolean padding mask, a ``True`` indicates that the
            corresponding key position is not allowed to attend.
        """
        embeds = self.embed(token_indices)

        if self.scale != 1.0:
            embeds = embeds * self.scale

        if self.pos_embed is not None:
            embeds = self.pos_embed(embeds, state_bag)

        if self.norm is not None:
            embeds = self.norm(embeds)

        if self.dropout_p > 0.0:
            embeds = F.dropout(embeds, self.dropout_p, self.training)

        if self.embed.padding_idx is None:
            padding_mask = None
        else:
            padding_mask = token_indices.eq(self.embed.padding_idx)

        return embeds, padding_mask

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"scale={self.scale}, dropout_p={self.dropout_p}"


class Transformer(Module):
    """Represents a Transformer model."""

    model_dim: int
    encoder_frontend: TransformerTokenFrontend
    encoder: TransformerEncoder
    decoder_frontend: TransformerTokenFrontend
    decoder: TransformerDecoder
    score_proj: Projection

    def __init__(
        self,
        encoder_frontend: TransformerTokenFrontend,
        encoder: TransformerEncoder,
        decoder_frontend: TransformerTokenFrontend,
        decoder: TransformerDecoder,
        score_proj: Projection,
    ) -> None:
        """
        :param encoder_frontend:
            The encoder front-end.
        :param encoder:
            The encoder.
        :param decoder_frontend:
            The decoder front-end.
        :param decoder:
            The decoder.
        :param score_proj:
            The projection to apply to outputs of the decoder.
        """
        model_dim = encoder.model_dim

        if decoder.model_dim != model_dim:
            raise ValueError(
                f"`model_dim` of `encoder` ({encoder.model_dim}) does not match `model_dim` of `decoder` ({decoder.model_dim})."
            )

        if encoder_frontend.embed.embedding_dim != model_dim:
            raise ValueError(
                f"`embedding_dim` of `encoder_frontend.embed` ({encoder_frontend.embed.embedding_dim}) does not match `model_dim` of `encoder` ({model_dim})."
            )

        if decoder_frontend.embed.embedding_dim != model_dim:
            raise ValueError(
                f"`embedding_dim` of `decoder_frontend.embed` ({decoder_frontend.embed.embedding_dim}) does not match `model_dim` of `decoder` ({model_dim})."
            )

        super().__init__()

        self.model_dim = model_dim

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder

        self.score_proj = score_proj

    def encode(self, token_indices: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Encode the specified source token indices.

        :param token_indices:
            The token indices to encode. *Shape:* :math:`(N,S)`, or :math:`(S)`
            when unbatched, where :math:`N` is the batch size and :math:`S` is
            the sequence length.

        :returns:
            - The encoded output. *Shape:* :math:`(N,S,M)`, or :math:`(S,M)`
              when unbatched, where :math:`N` is the batch size, :math:`S` is
              the sequence length, and :math:`M` is the model size.
            - The boolean padding mask indicating which key positions to ignore
              for the purpose of encoder-decoder attention. *Shape:*
              :math:`(N,S)`, or :math:`(S)` when unbatched, where :math:`N` is
              the batch size and :math:`S` is the sequence length.

        .. note::
            For a boolean padding mask, a ``True`` indicates that the
            corresponding key position is not allowed to attend.
        """
        embeds, padding_mask = self.encoder_frontend(token_indices)

        x = self.encoder(embeds)

        return x, padding_mask

    def decode_and_score(
        self,
        token_indices: Tensor,
        enc_out: Tensor,
        enc_padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        """Decode the specified target token indices.

        :param token_indices:
            The token indices to decode. *Shape:* :math:`(N,S)`, or :math:`(S)`
            when unbatched, where :math:`N` is the batch size and :math:`S` is
            the sequence length.
        :param enc_out:
            The encoder output for the encoder-decoder attention. *Shape:*
            :math:`(N,S_{src},M)`, or :math:`(S_{src},M)` when unbatched, where
            :math:`N` is the batch size, :math:`S_{src}` is the source sequence
            length, and :math:`M` is the model size.
        :param enc_padding_mask:
            The boolean or float padding mask indicating which key positions to
            ignore for the purpose of encoder-decoder attention. *Shape:*
            :math:`(N,S_{src})`, or :math:`(S_{src})` when unbatched, where
            :math:`N` is the batch size and :math:`S_{src}` is the source
            sequence length.
        :param state_bag:
            The state bag to use during an incremental evaluation.

        :returns:
            The output of :attr:`score_proj`. A softmax function should be
            applied to the produced scores to obtain the next-token
            probabilities. *Shape:* :math:`(N,S,D)`, or :math:`(S,D)` when
            unbatched, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`D` is the size of the output embedding
            dictionary.
        """
        embeds, padding_mask = self.decoder_frontend(token_indices, state_bag)

        x = self.decoder(embeds, padding_mask, enc_out, enc_padding_mask, state_bag)

        x = self.score_proj(x)

        return x  # type: ignore[no-any-return]

    def forward(self, src_token_indices: Tensor, tgt_token_indices: Tensor) -> Tensor:
        """
        :param src_token_indices:
            The source token indices to encode. *Shape:* :math:`(N,S_{src})`, or
            :math:`(S_{src})` when unbatched, where :math:`N` is the batch size
            and :math:`S_{src}` is the source sequence length.
        :param tgt_token_indices:
            The target token indices to decode. *Shape:* :math:`(N,S_{tgt})`, or
            :math:`(S_{tgt})` when unbatched, where :math:`N` is the batch size
            and :math:`S_{tgt}` is the target sequence length.

        :returns:
            The output of :attr:`score_proj`. A softmax function should be
            applied to the produced scores to obtain the next-token
            probabilities. *Shape:* :math:`(N,S_{tgt},D)`, or
            :math:`(S_{tgt},D)` when unbatched, where :math:`N` is the batch
            size, :math:`S_{tgt}` is the target sequence length, and :math:`D`
            is the size of the output embedding dictionary.
        """
        enc_out, enc_padding_mask = self.encode(src_token_indices)

        return self.decode_and_score(tgt_token_indices, enc_out, enc_padding_mask)


@final
class ScoreProjection(ResettableProjection):
    """Produces scores (i.e. logits) from the output of a Transformer decoder."""

    def __init__(
        self,
        num_embed: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param num_embed:
            The size of the output embedding dictionary.
        :param embedding_dim:
            The dimensionality of output embeddings.
        """
        super().__init__(
            embedding_dim, num_embed, bias=False, device=device, dtype=dtype
        )

    @finaloverride
    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        nn.init.normal_(self.weight, std=self.inp_dim**-0.5)

        if self.bias is not None:
            nn.init.zeros_(self.bias)
