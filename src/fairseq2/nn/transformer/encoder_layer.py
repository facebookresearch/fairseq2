# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Optional, Tuple, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Dropout, Module
from torch.nn.parameter import Parameter

from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer.attention_mask import AttentionMask
from fairseq2.nn.transformer.ffn import FeedForwardNetwork
from fairseq2.nn.transformer.layer_norm import (
    LayerNormFactory,
    create_standard_layer_norm,
)
from fairseq2.nn.transformer.multihead_attention import MultiheadAttention
from fairseq2.nn.transformer.norm_order import TransformerNormOrder
from fairseq2.typing import DataType, Device, finaloverride


class TransformerEncoderLayer(Module, ABC):
    """Represents a Transformer encoder layer."""

    model_dim: int

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        """
        super().__init__()

        self.model_dim = model_dim

    @abstractmethod
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        self_attn_mask: Optional[AttentionMask] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """
        :param seqs:
            The sequences to process. *Shape:* :math:`(N,S,M)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`M`
            is the dimensionality of the model.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.
        :param self_attn_mask:
            The mask that will be added to attention weights before computing
            the self attention. *Shape:* :math:`([H],S,S)`, where :math:`H` is
            the number of attention heads and :math:`S` is the sequence length.

        :returns:
            - The encoder layer output. *Shape:* Same as ``seqs``.
            - The padding mask of the encoder layer output. *Shape:* Same as
              ``padding_mask``.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


@final
class StandardTransformerEncoderLayer(TransformerEncoderLayer):
    """Represents a Transformer encoder layer as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`.
    """

    self_attn: MultiheadAttention
    self_attn_norm: Optional[LayerNorm]
    self_attn_dropout: Optional[Dropout]
    self_attn_layer_norm: LayerNorm
    ffn: FeedForwardNetwork
    ffn_dropout: Optional[Dropout]
    residual_scale: Optional[Parameter]
    ffn_layer_norm: LayerNorm
    norm_order: TransformerNormOrder

    def __init__(
        self,
        self_attn: MultiheadAttention,
        ffn: FeedForwardNetwork,
        *,
        scale_residual: bool = False,
        dropout_p: float = 0.1,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        layer_norm_factory: Optional[LayerNormFactory] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param self_attn:
            The self attention layer.
        :param ffn:
            The feed-forward network.
        :param scale_residual:
            If ``True``, scales residuals before adding them to the output of
            the feed-forward network as described in
            :cite:t:`https://doi.org/10.48550/arxiv.2110.09456`.
        :param dropout_p:
            The dropout probability on outputs of the self attention layer and
            the feed-forward network.
        :param norm_order:
            The Layer Normalization order.
        :param layer_norm_factory:
            The factory to construct the Layer Normalization modules.
        """
        model_dim = self_attn.model_dim

        super().__init__(model_dim)

        if layer_norm_factory is None:
            layer_norm_factory = create_standard_layer_norm

        self_attn_layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

        if norm_order != TransformerNormOrder.POST:
            self.self_attn_layer_norm = self_attn_layer_norm

        self.self_attn = self_attn

        if norm_order == TransformerNormOrder.PRE_WITH_NORMFORMER:
            self.self_attn_norm = layer_norm_factory(
                model_dim, device=device, dtype=dtype
            )
        else:
            self.register_module("self_attn_norm", None)

        if dropout_p > 0.0:
            self.self_attn_dropout = Dropout(dropout_p)
        else:
            self.register_module("self_attn_dropout", None)

        if norm_order == TransformerNormOrder.POST:
            self.self_attn_layer_norm = self_attn_layer_norm

        ffn_layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

        if norm_order != TransformerNormOrder.POST:
            self.ffn_layer_norm = ffn_layer_norm

        self.ffn = ffn

        if dropout_p > 0.0:
            self.ffn_dropout = Dropout(dropout_p)
        else:
            self.register_module("ffn_dropout", None)

        if scale_residual:
            self.residual_scale = Parameter(
                torch.empty((model_dim,), device=device, dtype=dtype)
            )
        else:
            self.register_parameter("residual_scale", None)

        if norm_order == TransformerNormOrder.POST:
            self.ffn_layer_norm = ffn_layer_norm

        self.norm_order = norm_order

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        if self.residual_scale is not None:
            nn.init.ones_(self.residual_scale)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        self_attn_mask: Optional[AttentionMask] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        seqs = self._forward_self_attn(seqs, padding_mask, self_attn_mask)

        seqs = self._forward_ffn(seqs)

        return seqs, padding_mask

    def _forward_self_attn(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        self_attn_mask: Optional[AttentionMask],
    ) -> Tensor:
        residual = seqs

        if self.norm_order != TransformerNormOrder.POST:
            seqs = self.self_attn_layer_norm(seqs)

        seqs = self.self_attn(
            seqs,
            padding_mask,
            keys=seqs,
            key_padding_mask=padding_mask,
            values=seqs,
            attn_mask=self_attn_mask,
        )

        if self.self_attn_norm is not None:
            seqs = self.self_attn_norm(seqs)

        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        seqs = seqs + residual

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.self_attn_layer_norm(seqs)

        return seqs

    def _forward_ffn(self, seqs: Tensor) -> Tensor:
        residual = seqs

        if self.norm_order != TransformerNormOrder.POST:
            seqs = self.ffn_layer_norm(seqs)

        seqs = self.ffn(seqs)

        if self.ffn_dropout is not None:
            seqs = self.ffn_dropout(seqs)

        if self.residual_scale is not None:
            residual = self.residual_scale * residual

        seqs = seqs + residual

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.ffn_layer_norm(seqs)

        return seqs

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, norm_order={self.norm_order}"
