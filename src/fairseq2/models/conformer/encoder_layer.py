# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn.functional as F
from overrides import final as finaloverride
from torch import Tensor
from torch.nn import LayerNorm

from fairseq2.models.conformer.convolution import ConformerConvolution
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    MultiheadAttention,
    TransformerEncoderLayer,
)


class ConformerEncoderLayer(TransformerEncoderLayer):
    """Represents a Conformer encoder layer as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2005.08100`."""

    ffn1_layer_norm: LayerNorm
    ffn1: FeedForwardNetwork
    self_attn_layer_norm: LayerNorm
    self_attn: MultiheadAttention
    conv_layer_norm: LayerNorm
    conv: ConformerConvolution
    ffn2_layer_norm: LayerNorm
    ffn2: FeedForwardNetwork
    layer_norm: LayerNorm
    dropout_p: float

    def __init__(
        self,
        ffn1: FeedForwardNetwork,
        self_attn: MultiheadAttention,
        conv: ConformerConvolution,
        ffn2: FeedForwardNetwork,
        dropout_p: float = 0.1,
        norm_eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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
        :param norm_eps:
            The epsilon value to add to the denominator of the
            :class:`~torch.nn.LayerNorm` modules for numerical stability.
        """
        model_dim = self_attn.model_dim

        super().__init__(model_dim)

        if ffn1.model_dim != model_dim:
            raise ValueError(
                f"`model_dim` of `ffn1` ({ffn1.model_dim}) does not match `model_dim` of `self_attn` ({model_dim})."
            )

        self.ffn1_layer_norm = LayerNorm(
            model_dim, norm_eps, device=device, dtype=dtype
        )

        self.ffn1 = ffn1

        self.self_attn_layer_norm = LayerNorm(
            model_dim, norm_eps, device=device, dtype=dtype
        )

        self.self_attn = self_attn

        if conv.model_dim != model_dim:
            raise ValueError(
                f"`model_dim` of `conv` ({conv.model_dim}) does not match `model_dim` of `self_attn` ({model_dim})."
            )

        self.conv_layer_norm = LayerNorm(
            model_dim, norm_eps, device=device, dtype=dtype
        )

        self.conv = conv

        if ffn2.model_dim != model_dim:
            raise ValueError(
                f"`model_dim` of `ffn2` ({ffn2.model_dim}) does not match `model_dim` of `self_attn` ({model_dim})."
            )

        self.ffn2_layer_norm = LayerNorm(
            model_dim, norm_eps, device=device, dtype=dtype
        )

        self.ffn2 = ffn2

        self.layer_norm = LayerNorm(model_dim, norm_eps, device=device, dtype=dtype)

        self.dropout_p = dropout_p

    @finaloverride
    def forward(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self._forward_ffn1(x)

        x = self._forward_self_attn(x, padding_mask, self_attn_mask)

        x = self._forward_conv(x)

        x = self._forward_ffn2(x)

        x = self.layer_norm(x)

        return x  # type: ignore[no-any-return]

    def _forward_ffn1(self, x: Tensor) -> Tensor:
        residual = x

        x = self.ffn1_layer_norm(x)

        x = self.ffn1(x) * 0.5

        if self.dropout_p > 0.0:
            x = F.dropout(x, self.dropout_p, self.training)

        return x + residual

    def _forward_self_attn(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor],
        self_attn_mask: Optional[Tensor],
    ) -> Tensor:
        residual = x

        x = self.self_attn_layer_norm(x)

        x = self.self_attn(
            x,
            keys=x,
            values=x,
            attn_mask=self_attn_mask,
            padding_mask=padding_mask,
        )

        if self.dropout_p > 0.0:
            x = F.dropout(x, self.dropout_p, self.training)

        return x + residual

    def _forward_conv(self, x: Tensor) -> Tensor:
        residual = x

        x = self.conv_layer_norm(x)

        x = self.conv(x)

        if self.dropout_p > 0.0:
            x = F.dropout(x, self.dropout_p, self.training)

        return x + residual

    def _forward_ffn2(self, x: Tensor) -> Tensor:
        residual = x

        x = self.ffn2_layer_norm(x)

        x = self.ffn2(x) * 0.5

        if self.dropout_p > 0.0:
            x = F.dropout(x, self.dropout_p, self.training)

        return x + residual

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, dropout_p={self.dropout_p}"
