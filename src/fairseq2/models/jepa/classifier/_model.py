# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter

from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.transformer import TransformerFrontend
from fairseq2.nn import LayerNorm, Projection
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    LayerNormFactory,
    MultiheadAttention,
    TransformerEncoder,
    create_standard_layer_norm,
)
from fairseq2.typing import DataType, Device


@final
class JepaClassifierModel(Module):
    """
    Represents a pretrained Jepa model, with an attentive probing layer for
    classfication tasks. See
        * :cite:t:`https://doi.org/10.48550/arXiv.2301.08243`
        * :cite:t:`https://doi.org/10.48550/arXiv.2404.08471`
    """

    model_dim: int
    encoder_frontend: TransformerFrontend
    encoder: TransformerEncoder
    pooler: AttentivePooler
    head: Projection

    def __init__(
        self,
        encoder_frontend: TransformerFrontend,
        encoder: TransformerEncoder,
        pooler: AttentivePooler,
        head: Projection,
    ) -> None:
        super().__init__()

        self.model_dim = encoder.model_dim

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder

        self.pooler = pooler

        self.head = head

    def forward(self, batch: SequenceBatch) -> Tensor:
        seqs, padding_mask = self.encoder_frontend(batch.seqs, batch.padding_mask)

        seqs, _ = self.encoder(seqs, padding_mask)

        seqs = self.pooler(seqs)

        # (N, P, M)
        seqs = seqs.squeeze(1)  # TODO: NEEDED?

        return self.head(seqs)  # type: ignore[no-any-return]

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


@final
class AttentivePooler(Module):
    """
    An attentive pooler that gets output of a Jepa encoder and decode it into
    a logit of a given task.

    TODO:
    - Move this into fairseq2.nn to benefit other similiar tasks. Internally,
    this module is just a thin transformer encoder without self attention layer.
    Optionally, it can consist of some extra transformer encoders depending on the
    (finetuning) task
    """

    model_dim: int
    decoder: CrossAttentionDecoderLayer
    encoder: TransformerEncoder | None
    query_tokens: Parameter
    init_std: float

    def __init__(
        self,
        decoder: CrossAttentionDecoderLayer,
        encoder: TransformerEncoder | None,
        *,
        num_queries: int = 1,
        init_std: float = 0.02,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        self.model_dim = decoder.model_dim

        self.decoder = decoder

        if encoder:
            self.encoder = encoder
        else:
            self.register_module("encoder", None)

        self.query_tokens = Parameter(
            torch.empty((1, num_queries, self.model_dim), device=device, dtype=dtype)
        )

        self.init_std = init_std

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        nn.init.trunc_normal_(self.query_tokens, std=self.init_std)

    def forward(self, seqs: Tensor) -> Tensor:
        if self.encoder is not None:
            seqs, _ = self.encoder(seqs, padding_mask=None)

        batch_size = seqs.size(0)

        # (1, P, M) -> (N, P, M)
        pool_seqs = self.query_tokens.repeat(batch_size, 1, 1)

        return self.decoder(pool_seqs, seqs)  # type: ignore[no-any-return]

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}, num_queries={self.query_tokens.size(1)}"


@final
class CrossAttentionDecoderLayer(Module):
    """Represents a simple transformer decoder with only cross attention and layernorm"""

    model_dim: int
    cross_attn_layer_norm: LayerNorm
    cross_attn: MultiheadAttention
    ffn_layer_norm: LayerNorm
    ffn: FeedForwardNetwork

    def __init__(
        self,
        cross_attn: MultiheadAttention,
        ffn: FeedForwardNetwork,
        *,
        layer_norm_factory: LayerNormFactory | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param cross_attn:
            The encoder-decoder attention layer.
        :param ffn:
            The feed-forward network.
        :param layer_norm_factory:
            The factory to construct the Layer Normalization modules.
        """
        super().__init__()

        model_dim = cross_attn.model_dim

        if layer_norm_factory is None:
            layer_norm_factory = create_standard_layer_norm

        self.cross_attn_layer_norm = layer_norm_factory(
            model_dim, device=device, dtype=dtype
        )

        self.model_dim = model_dim

        self.cross_attn = cross_attn

        self.ffn_layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

        self.ffn = ffn

    def forward(self, seqs: Tensor, encoder_output: Tensor) -> Tensor:
        seqs = self._forward_cross_attn(seqs, encoder_output)

        seqs = self._forward_ffn(seqs)

        return seqs

    def _forward_cross_attn(self, seqs: Tensor, encoder_output: Tensor) -> Tensor:
        residual = seqs

        # Note that the cross-attention norm is applied on encoder output and not seqs
        encoder_output = self.cross_attn_layer_norm(encoder_output)

        seqs = self.cross_attn(
            seqs,
            padding_mask=None,
            keys=encoder_output,
            key_padding_mask=None,
            values=encoder_output,
        )

        seqs = seqs + residual

        return seqs

    def _forward_ffn(self, seqs: Tensor) -> Tensor:
        residual = seqs

        seqs = self.ffn_layer_norm(seqs)

        seqs = self.ffn(seqs)

        seqs = seqs + residual

        return seqs

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"
