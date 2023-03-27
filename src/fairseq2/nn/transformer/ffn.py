# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Callable, Optional, final

import torch
import torch.nn.functional as F
from overrides import final as finaloverride
from torch import Tensor
from torch.nn import LayerNorm, Module

from fairseq2.nn.projection import Linear
from fairseq2.nn.transformer.norm_order import TransformerNormOrder
from fairseq2.nn.utils.fn import get_name


class FeedForwardNetwork(Module, ABC):
    """Represents a Transformer feed-forward network."""

    model_dim: int

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model (i.e. inputs and outputs).
        """
        super().__init__()

        self.model_dim = model_dim

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        :param x:
            The input to project. *Shape:* :math:`(N,S,M)`, or :math:`(S,M)`
            when unbatched, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`M` is the model size.

        :returns:
            The projected output. *Shape:* Same as ``x``.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


@final
class StandardFeedForwardNetwork(FeedForwardNetwork):
    """Represents a Transformer feed-forward network as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    inner_proj: Linear
    inner_activation_fn: Callable[[Tensor], Tensor]
    inner_dropout_p: float
    inner_norm: Optional[LayerNorm]
    out_proj: Linear

    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        inner_activation_fn: Optional[Callable[[Tensor], Tensor]] = None,
        inner_dropout_p: float = 0.0,
        bias: bool = True,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        norm_eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model (i.e. inputs and outputs).
        :param inner_dim:
            The dimensionality of the inner projection layer.
        :param inner_activation_fn:
            The activation to apply to outputs of the inner projection layer. If
            ``None``, :func:`~torch.nn.functional.relu` will be used.
        :param inner_dropout_p:
            The dropout probability on outputs of the inner projection layer.
        :param bias:
            If ``True``, both the inner and output projection layers will learn
            an additive bias.
        :param norm_order:
            The Layer Normalization order to use.
        :param norm_eps:
            When ``norm_order`` is ``PRE_WITH_NORMFORMER``, the epsilon value to
            add to the denominator of the :class:`~torch.nn.LayerNorm` module
            for numerical stability.
        """
        super().__init__(model_dim)

        self.inner_proj = Linear(
            model_dim, inner_dim, bias=bias, device=device, dtype=dtype
        )

        if inner_activation_fn is None:
            self.inner_activation_fn = F.relu
        else:
            self.inner_activation_fn = inner_activation_fn

        self.inner_dropout_p = inner_dropout_p

        if norm_order == TransformerNormOrder.PRE_WITH_NORMFORMER:
            self.inner_norm = LayerNorm(inner_dim, norm_eps, device=device, dtype=dtype)
        else:
            self.register_module("inner_norm", None)

        self.out_proj = Linear(
            inner_dim, model_dim, bias=bias, device=device, dtype=dtype
        )

    @finaloverride
    def forward(self, x: Tensor) -> Tensor:
        x = self.inner_proj(x)

        x = self.inner_activation_fn(x)

        if self.inner_norm is not None:
            x = self.inner_norm(x)

        if self.inner_dropout_p > 0.0:
            x = F.dropout(x, self.inner_dropout_p, self.training)

        x = self.out_proj(x)

        return x

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, inner_activation_fn={get_name(self.inner_activation_fn)}, inner_dropout_p={self.inner_dropout_p}"
