# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch import Tensor
from torch.nn import Dropout
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.conformer.convolution import ConformerConvolution
from fairseq2.models.transformer import (
    AttentionBiasCache,
    FeedForwardNetwork,
    MultiheadAttention,
    TransformerEncoderLayer,
)
from fairseq2.nn import BatchLayout, LayerNorm


@final
class ConformerBlock(TransformerEncoderLayer):
    """Represents a Conformer block as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2005.08100`."""

    def __init__(
        self,
        ffn1_layer_norm: LayerNorm,
        ffn1: FeedForwardNetwork,
        self_attn_layer_norm: LayerNorm,
        self_attn: MultiheadAttention,
        conv_layer_norm: LayerNorm,
        conv: ConformerConvolution,
        ffn2_layer_norm: LayerNorm,
        ffn2: FeedForwardNetwork,
        layer_norm: LayerNorm,
        *,
        dropout_p: float = 0.0,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param ffn1:
            The bottom macaron-like feed-forward network.
        :param self_attn:
            The self attention layer.
        :param conv:
            The Conformer convolution module.
        :param ffn2:
            The top macaron-like feed-forward network.
        :param dropout_p:
            The dropout probability on outputs of the self attention layer, the
            feed-forward networks, and the Conformer convolution module.
        :param layer_norm_factory:
            The factory to construct the Layer Normalization modules.
        """
        super().__init__()

        self.ffn1_layer_norm = ffn1_layer_norm

        self.ffn1 = ffn1

        if dropout_p > 0.0:
            ffn1_dropout = Dropout(dropout_p)
        else:
            ffn1_dropout = None

        self.ffn1_dropout: Dropout | None

        self.register_module("ffn1_dropout", ffn1_dropout)

        self.self_attn_layer_norm = self_attn_layer_norm

        self.self_attn = self_attn

        if dropout_p > 0.0:
            self_attn_dropout = Dropout(dropout_p)
        else:
            self_attn_dropout = None

        self.self_attn_dropout: Dropout | None

        self.register_module("self_attn_dropout", self_attn_dropout)

        self.conv_layer_norm = conv_layer_norm

        self.conv = conv

        if dropout_p > 0.0:
            conv_dropout = Dropout(dropout_p)
        else:
            conv_dropout = None

        self.conv_dropout: Dropout | None

        self.register_module("conv_dropout", conv_dropout)

        self.ffn2_layer_norm = ffn2_layer_norm

        self.ffn2 = ffn2

        if dropout_p > 0.0:
            ffn2_dropout = Dropout(dropout_p)
        else:
            ffn2_dropout = None

        self.ffn2_dropout: Dropout | None

        self.register_module("ffn2_dropout", ffn2_dropout)

        self.layer_norm = layer_norm

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        attn_bias_cache: AttentionBiasCache,
    ) -> Tensor:
        seqs = self._forward_ffn1(seqs)

        seqs = self._forward_self_attn(seqs, seqs_layout, attn_bias_cache)

        seqs = self._forward_conv(seqs, seqs_layout)

        seqs = self._forward_ffn2(seqs)

        seqs = self.layer_norm(seqs)

        return seqs

    def _forward_ffn1(self, seqs: Tensor) -> Tensor:
        residual = seqs

        seqs = self.ffn1_layer_norm(seqs)

        seqs = self.ffn1(seqs) * 0.5

        if self.ffn1_dropout is not None:
            seqs = self.ffn1_dropout(seqs)

        return seqs + residual

    def _forward_self_attn(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        attn_bias_cache: AttentionBiasCache,
    ) -> Tensor:
        residual = seqs

        seqs = self.self_attn_layer_norm(seqs)

        seqs = self.self_attn(
            seqs,
            seqs_layout,
            keys=seqs,
            keys_layout=seqs_layout,
            values=seqs,
            bias_cache=attn_bias_cache,
        )

        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        return seqs + residual

    def _forward_conv(self, seqs: Tensor, seqs_layout: BatchLayout) -> Tensor:
        residual = seqs

        seqs = self.conv_layer_norm(seqs)

        seqs = self.conv(seqs, seqs_layout)

        if self.conv_dropout is not None:
            seqs = self.conv_dropout(seqs)

        return seqs + residual

    def _forward_ffn2(self, seqs: Tensor) -> Tensor:
        residual = seqs

        seqs = self.ffn2_layer_norm(seqs)

        seqs = self.ffn2(seqs) * 0.5

        if self.ffn2_dropout is not None:
            seqs = self.ffn2_dropout(seqs)

        return seqs + residual
