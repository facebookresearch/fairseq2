# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, final

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import final as finaloverride
from torch import Tensor
from torch.nn import LayerNorm, Module
from torch.nn.parameter import Parameter

from fairseq2.nn.transformer.ffn import FeedForwardNetwork
from fairseq2.nn.transformer.multihead_attention import MultiheadAttention
from fairseq2.nn.transformer.norm_order import TransformerNormOrder
from fairseq2.typing import DataType, Device


class TransformerEncoderLayer(Module, ABC):
    """Represents a Transformer encoder layer."""

    model_dim: int
    """The dimensionality of the model (i.e. inputs and outputs)."""

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model (i.e. inputs and outputs).
        """
        super().__init__()

        self.model_dim = model_dim

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        :param x:
            The input to process. *Shape:* :math:`(N,S,M)`, or :math:`(S,M)`
            when unbatched, where :math:`N` is the batch size, :math:`S` is the
            source sequence length, and :math:`M` is the model size.
        :param self_attn_mask:
            The float mask that will be added to the attention weights before
            computing the self attention. *Shape:* :math:`(S,S)`, where
            :math:`S` is the source sequence length.
        :param self_attn_padding_mask:
            The boolean or float key padding mask indicating which key positions
            to ignore for the purpose of self attention. *Shape:* :math:`(N,S)`,
            or :math:`(S)` when unbatched, where :math:`N` is the batch size and
            :math:`S` is the source sequence length.

        :returns:
            The output. *Shape:* Same as ``x``.

        .. note::
            For a boolean key padding mask, a ``True`` indicates that the
            corresponding key position is not allowed to attend. For a float key
            padding mask, the mask values will be added to the attention
            weights.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


@final
class StandardTransformerEncoderLayer(TransformerEncoderLayer):
    """Represents a Transformer encoder layer as described in
    :cite:t:`DBLP:journals/corr/VaswaniSPUJGKP17`.
    """

    self_attn: MultiheadAttention
    self_attn_norm: Optional[LayerNorm]
    self_attn_layer_norm: LayerNorm
    ffn: FeedForwardNetwork
    residual_scale: Optional[Parameter]
    ffn_layer_norm: LayerNorm
    dropout_p: float
    norm_order: TransformerNormOrder

    def __init__(
        self,
        self_attn: MultiheadAttention,
        ffn: FeedForwardNetwork,
        scale_residual: bool = False,
        dropout_p: float = 0.1,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        norm_eps: float = 1e-5,
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
            the feed-forward network. See
            :cite:t:`DBLP:journals/corr/abs-2110-09456` for more information.
        :param dropout_p:
            The dropout probability on the outputs of the self attention layer
            and the feed-forward network.
        :param norm_order:
            The Layer Normalization order to use.
        :param norm_eps:
            The epsilon value to add to the denominator of the
            :class:`~torch.nn.LayerNorm` modules for numerical stability.
        """
        fct_kwargs: Dict[str, Any] = {"device": device, "dtype": dtype}

        model_dim = self_attn.model_dim

        super().__init__(model_dim)

        self_attn_layer_norm = LayerNorm(model_dim, norm_eps, **fct_kwargs)

        if norm_order != TransformerNormOrder.POST:
            self.self_attn_layer_norm = self_attn_layer_norm

        self.self_attn = self_attn

        if norm_order == TransformerNormOrder.PRE_WITH_NORMFORMER:
            self.self_attn_norm = LayerNorm(model_dim, norm_eps, **fct_kwargs)
        else:
            self.register_module("self_attn_norm", None)

        if norm_order == TransformerNormOrder.POST:
            self.self_attn_layer_norm = self_attn_layer_norm

        if ffn.model_dim != model_dim:
            raise ValueError(
                f"`model_dim` of `ffn` ({ffn.model_dim}) does not match `model_dim` ({model_dim})."
            )

        ffn_layer_norm = LayerNorm(model_dim, norm_eps, **fct_kwargs)

        if norm_order != TransformerNormOrder.POST:
            self.ffn_layer_norm = ffn_layer_norm

        self.ffn = ffn

        if scale_residual:
            self.residual_scale = Parameter(torch.empty((model_dim,), **fct_kwargs))
        else:
            self.register_parameter("residual_scale", None)

        if norm_order == TransformerNormOrder.POST:
            self.ffn_layer_norm = ffn_layer_norm

        self.dropout_p = dropout_p

        self.norm_order = norm_order

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters and buffers of the module."""
        if self.residual_scale is not None:
            nn.init.ones_(self.residual_scale)

    @finaloverride
    def forward(
        self,
        x: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self._forward_self_attn(x, self_attn_mask, self_attn_padding_mask)

        x = self._forward_ffn(x)

        return x

    def _forward_self_attn(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        attn_padding_mask: Optional[Tensor],
    ) -> Tensor:
        residual = x

        if self.norm_order != TransformerNormOrder.POST:
            x = self.self_attn_layer_norm(x)

        x = self.self_attn(
            x,
            keys=x,
            values=x,
            attn_mask=attn_mask,
            padding_mask=attn_padding_mask,
        )

        if self.self_attn_norm is not None:
            x = self.self_attn_norm(x)

        if self.dropout_p > 0.0:
            x = F.dropout(x, self.dropout_p, self.training)

        x = x + residual

        if self.norm_order == TransformerNormOrder.POST:
            x = self.self_attn_layer_norm(x)

        return x

    def _forward_ffn(self, x: Tensor) -> Tensor:
        residual = x

        if self.norm_order != TransformerNormOrder.POST:
            x = self.ffn_layer_norm(x)

        x = self.ffn(x)

        if self.dropout_p > 0.0:
            x = F.dropout(x, self.dropout_p, self.training)

        if self.residual_scale is not None:
            residual = torch.mul(self.residual_scale, residual)

        x = x + residual

        if self.norm_order == TransformerNormOrder.POST:
            x = self.ffn_layer_norm(x)

        return x
