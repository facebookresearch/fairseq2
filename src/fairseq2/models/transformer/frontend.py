# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple, final

import torch
from overrides import final as finaloverride
from torch import Tensor
from torch.nn import Dropout, LayerNorm

from fairseq2.models.encoder_decoder import DecoderFrontend
from fairseq2.nn.embedding import Embedding
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.positional_encoder import PositionalEncoder
from fairseq2.nn.utils.mask import to_padding_mask


@final
class TransformerFrontend(DecoderFrontend):
    """Represents a Transformer encoder/decoder front-end as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    embed: Embedding
    scale: float
    pos_encoder: Optional[PositionalEncoder]
    layer_norm: Optional[LayerNorm]
    dropout: Optional[Dropout]

    def __init__(
        self,
        embed: Embedding,
        pos_encoder: Optional[PositionalEncoder],
        no_scale: bool = False,
        use_layer_norm: bool = False,
        dropout_p: float = 0.1,
        norm_eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param embed:
            The token embedding table.
        :param pos_encoder:
            The positional encoder.
        :param no_scale:
            If ``True``, does not scale embeddings by the square root of the
            embedding size.
        :param use_layer_norm:
            If ``True``, applies Layer Normalization to embeddings.
        :param dropout_p:
            The dropout probability on embeddings.
        :param norm_eps:
            The epsilon value to add to the denominator of the
            :class:`~torch.nn.LayerNorm` module for numerical stability.
        """
        model_dim = embed.embedding_dim

        super().__init__(model_dim)

        self.embed = embed

        self.scale = 1.0 if no_scale else math.sqrt(model_dim)

        if pos_encoder is not None:
            if pos_encoder.dim != model_dim:
                raise ValueError(
                    f"`dim` of `pos_encoder` and `embedding_dim` of `embed` must be equal, but are {pos_encoder.dim} and {model_dim} instead."
                )

            self.pos_encoder = pos_encoder
        else:
            self.register_module("pos_encoder", None)

        if use_layer_norm:
            self.layer_norm = LayerNorm(model_dim, norm_eps, device=device, dtype=dtype)
        else:
            self.register_module("layer_norm", None)

        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        seqs = self.embed(seqs)

        padding_mask = to_padding_mask(seqs, seq_lens)

        if self.scale != 1.0:
            seqs = seqs * self.scale

        if self.pos_encoder is not None:
            seqs = self.pos_encoder(seqs, padding_mask, state_bag)

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        return seqs, padding_mask

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return s + ", no_scale=False" if self.scale != 1.0 else ""
