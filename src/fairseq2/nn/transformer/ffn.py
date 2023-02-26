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
            sequence length, and :math:`M` is the batch size.

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
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model (i.e. inputs and outputs).
        :param inner_dim:
            The dimensionality of the inner layer.
        :param inner_activation_fn:
            The activation to apply to outputs of the inner layer. If ``None``,
            :func:`~torch.nn.functional.relu` will be used.
        :param inner_dropout_p:
            The dropout probability on outputs of the inner layer.
        :param norm_order:
            The Layer Normalization order to use.
        """
        super().__init__(model_dim)

        self.inner_proj = Linear(
            model_dim, inner_dim, bias=True, device=device, dtype=dtype
        )

        if inner_activation_fn is None:
            self.inner_activation_fn = F.relu
        else:
            self.inner_activation_fn = inner_activation_fn

        self.inner_dropout_p = inner_dropout_p

        if norm_order == TransformerNormOrder.PRE_WITH_NORMFORMER:
            self.inner_norm = LayerNorm(inner_dim, device=device, dtype=dtype)
        else:
            self.register_module("inner_norm", None)

        self.out_proj = Linear(
            inner_dim, model_dim, bias=True, device=device, dtype=dtype
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

        return f"{s}, inner_dropout_p={self.inner_dropout_p}"
