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

from fairseq2.models.encoder_decoder import EncoderDecoderFrontend
from fairseq2.nn.embedding import Embedding
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.positional_embedding import PositionalEmbedding


@final
class TransformerTokenFrontend(EncoderDecoderFrontend):
    """Represents a Transformer model front-end as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    embed: Embedding
    scale: float
    pos_embed: Optional[PositionalEmbedding]
    layer_norm: Optional[LayerNorm]
    dropout: Optional[Dropout]

    def __init__(
        self,
        embed: Embedding,
        pos_embed: Optional[PositionalEmbedding],
        no_scale: bool = False,
        layer_norm: bool = False,
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
            If ``True``, does not scale embeddings by the square root of the
            embedding size.
        :param layer_norm:
            If ``True``, applies Layer Normalization to embeddings.
        :param dropout_p:
            The dropout probability on outputs.
        :param norm_eps:
            The epsilon value to add to the denominator of the
            :class:`~torch.nn.LayerNorm` module for numerical stability.
        """
        model_dim = embed.embed_dim

        super().__init__(model_dim)

        self.embed = embed

        self.scale = 1.0 if no_scale else math.sqrt(model_dim)

        if pos_embed is not None:
            if pos_embed.embed_dim != model_dim:
                raise ValueError(
                    f"`embed_dim` of `pos_embed` and `embed_dim` of `embed` must be equal, but are {pos_embed.embed_dim} and {model_dim} instead."
                )

            self.pos_embed = pos_embed
        else:
            self.register_module("pos_embed", None)

        if layer_norm:
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
        x = self.embed(seqs)

        if self.scale != 1.0:
            x = x * self.scale

        if self.pos_embed is not None:
            x = self.pos_embed(x, state_bag)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x, seq_lens

    def extra_repr(self) -> str:
        """:meta private:"""
        return "no_scale=False" if self.scale != 1.0 else ""
