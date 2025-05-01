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

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.transformer import (
    FeedForwardNetwork,
    LayerNormFactory,
    MultiheadAttention,
    TransformerEncoder,
    TransformerFrontend,
    create_standard_layer_norm,
)
from fairseq2.nn import BatchLayout, LayerNorm, Projection


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
    attn_pooler: AttentivePooler
    head_proj: Projection

    def __init__(
        self,
        encoder_frontend: TransformerFrontend,
        encoder: TransformerEncoder,
        attn_pooler: AttentivePooler,
        head_proj: Projection,
    ) -> None:
        super().__init__()

        self.model_dim = encoder.model_dim

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder

        self.attn_pooler = attn_pooler

        self.head_proj = head_proj

    def forward(self, seqs: Tensor, seqs_layout: BatchLayout) -> Tensor:
        seqs, seqs_layout = self.encoder_frontend(seqs, seqs_layout)

        seqs = self.encoder(seqs, seqs_layout)

        seqs = self.attn_pooler(seqs, seqs_layout)

        # (N, P, M)
        seqs = seqs.squeeze(1)  # TODO: NEEDED?

        return self.head_proj(seqs)  # type: ignore[no-any-return]

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
    decoder_layer: CrossAttentionDecoderLayer
    encoder: TransformerEncoder | None
    query_tokens: Parameter
    init_std: float

    def __init__(
        self,
        decoder_layer: CrossAttentionDecoderLayer,
        encoder: TransformerEncoder | None,
        *,
        num_queries: int = 1,
        init_std: float = 0.02,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        self.model_dim = decoder_layer.model_dim

        self.decoder_layer = decoder_layer

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

    def forward(self, seqs: Tensor, seqs_layout: BatchLayout) -> Tensor:
        if self.encoder is not None:
            seqs = self.encoder(seqs, seqs_layout)

        batch_size = seqs.size(0)

        # (1, P, M) -> (N, P, M)
        pool_seqs = self.query_tokens.repeat(batch_size, 1, 1)

        pool_seqs_layout = BatchLayout.of(pool_seqs)

        return self.decoder_layer(  # type: ignore[no-any-return]
            pool_seqs, pool_seqs_layout, seqs, seqs_layout
        )

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

    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        encoder_output: Tensor,
        encoder_output_layout: BatchLayout,
    ) -> Tensor:
        seqs = self._forward_cross_attn(
            seqs, seqs_layout, encoder_output, encoder_output_layout
        )

        seqs = self._forward_ffn(seqs)

        return seqs

    def _forward_cross_attn(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        encoder_output: Tensor,
        encoder_output_layout: BatchLayout,
    ) -> Tensor:
        residual = seqs

        # Note that the cross-attention norm is applied on encoder output instead
        # of sequences.
        encoder_output = self.cross_attn_layer_norm(encoder_output)

        seqs = self.cross_attn(
            seqs,
            seqs_layout,
            keys=encoder_output,
            keys_layout=encoder_output_layout,
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
