# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

from torch import Tensor
from torch.nn import Dropout

from fairseq2.models.conformer.convolution import ConformerConvolution
from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer import (
    AttentionMask,
    FeedForwardNetwork,
    LayerNormFactory,
    MultiheadAttention,
    TransformerEncoderLayer,
    create_standard_layer_norm,
)
from fairseq2.typing import DataType, Device, finaloverride


class ConformerBlock(TransformerEncoderLayer):
    """Represents a Conformer block as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2005.08100`."""

    ffn1_layer_norm: LayerNorm
    ffn1: FeedForwardNetwork
    ffn1_dropout: Optional[Dropout]
    self_attn_layer_norm: LayerNorm
    self_attn: MultiheadAttention
    self_attn_dropout: Optional[Dropout]
    conv_layer_norm: LayerNorm
    conv: ConformerConvolution
    conv_dropout: Optional[Dropout]
    ffn2_layer_norm: LayerNorm
    ffn2: FeedForwardNetwork
    ffn2_dropout: Optional[Dropout]
    layer_norm: LayerNorm

    def __init__(
        self,
        ffn1: FeedForwardNetwork,
        self_attn: MultiheadAttention,
        conv: ConformerConvolution,
        ffn2: FeedForwardNetwork,
        *,
        dropout_p: float = 0.1,
        layer_norm_factory: Optional[LayerNormFactory] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
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
        model_dim = self_attn.model_dim

        super().__init__(model_dim)

        if layer_norm_factory is None:
            layer_norm_factory = create_standard_layer_norm

        self.ffn1_layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

        self.ffn1 = ffn1

        if dropout_p > 0.0:
            self.ffn1_dropout = Dropout(dropout_p)
        else:
            self.register_module("ffn1_dropout", None)

        self.self_attn_layer_norm = layer_norm_factory(
            model_dim, device=device, dtype=dtype
        )

        self.self_attn = self_attn

        if dropout_p > 0.0:
            self.self_attn_dropout = Dropout(dropout_p)
        else:
            self.register_module("self_attn_dropout", None)

        self.conv_layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

        self.conv = conv

        if dropout_p > 0.0:
            self.conv_dropout = Dropout(dropout_p)
        else:
            self.register_module("conv_dropout", None)

        self.ffn2_layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

        self.ffn2 = ffn2

        if dropout_p > 0.0:
            self.ffn2_dropout = Dropout(dropout_p)
        else:
            self.register_module("ffn2_dropout", None)

        self.layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        self_attn_mask: Optional[AttentionMask] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        seqs = self._forward_ffn1(seqs)

        seqs = self._forward_self_attn(seqs, padding_mask, self_attn_mask)

        seqs = self._forward_conv(seqs, padding_mask)

        seqs = self._forward_ffn2(seqs)

        seqs = self.layer_norm(seqs)

        return seqs, padding_mask

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
        padding_mask: Optional[PaddingMask],
        self_attn_mask: Optional[AttentionMask],
    ) -> Tensor:
        residual = seqs

        seqs = self.self_attn_layer_norm(seqs)

        seqs = self.self_attn(
            seqs,
            padding_mask,
            keys=seqs,
            key_padding_mask=padding_mask,
            values=seqs,
            attn_mask=self_attn_mask,
        )

        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        return seqs + residual

    def _forward_conv(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tensor:
        residual = seqs

        seqs = self.conv_layer_norm(seqs)

        seqs = self.conv(seqs, padding_mask)

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
