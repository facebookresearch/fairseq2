# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, final

import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Module

from ..projection import LocalProjection
from .norm_order import TransformerNormOrder


class FeedForwardNetwork(Module, ABC):
    """Represents a Transformer feed-forward network.

    :param model_dim:
        The dimensionality of the model (i.e. inputs and outputs).
    """

    model_dim: int
    """The dimensionality of the model (i.e. inputs and outputs)."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()

        self.model_dim = model_dim

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        :param x:
            The input to process. *Shape:* :math:`(*,M)`, where :math:`M` is the
            model size.

        :returns:
            The output. *Shape:* Same as the input.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


@final
class StandardFeedForwardNetwork(FeedForwardNetwork):
    """Represents a Transformer feed-forward network as described in
    :cite:t:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    :param model_dim:
        The dimensionality of the model (i.e. inputs and outputs).
    :param inner_dim:
        The dimensionality of the inner layer.
    :param inner_activation_fn:
        The activation to apply to the output of the inner layer. If ``None``,
        :func:`~torch.nn.functional.relu` will be used.
    :param inner_dropout_p:
        The dropout probability on the output of the inner layer.
    :param norm_order:
        The Layer Normalization order to use.
    """

    inner_proj: LocalProjection
    inner_activation_fn: Callable[[Tensor], Tensor]
    inner_dropout_p: float
    inner_norm: Optional[LayerNorm]
    out_proj: LocalProjection

    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        inner_activation_fn: Optional[Callable[[Tensor], Tensor]] = None,
        inner_dropout_p: float = 0.0,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        device=None,
        dtype=None,
    ) -> None:
        fct_kwargs: Dict = {"device": device, "dtype": dtype}

        super().__init__(model_dim)

        self.inner_proj = LocalProjection(model_dim, inner_dim, **fct_kwargs)

        if inner_activation_fn is None:
            self.inner_activation_fn = F.relu  # type: ignore
        else:
            self.inner_activation_fn = inner_activation_fn  # type: ignore

        self.inner_dropout_p = inner_dropout_p

        if norm_order == TransformerNormOrder.PRE_WITH_NORMFORMER:
            self.inner_norm = LayerNorm(inner_dim, **fct_kwargs)
        else:
            self.register_module("inner_norm", None)

        self.out_proj = LocalProjection(inner_dim, model_dim, **fct_kwargs)

    def forward(self, x: Tensor) -> Tensor:  # override
        x = self.inner_proj(x)

        x = self.inner_activation_fn(x)  # type: ignore

        if self.inner_norm is not None:
            x = self.inner_norm(x)

        if self.inner_dropout_p > 0.0:
            x = F.dropout(x, self.inner_dropout_p, self.training)

        return self.out_proj(x)
