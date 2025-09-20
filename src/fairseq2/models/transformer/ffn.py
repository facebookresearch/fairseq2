# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, final

from torch import Tensor
from torch.nn import Dropout, Module, ReLU, Sigmoid, SiLU
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.nn import Linear


class FeedForwardNetwork(Module, ABC):
    """Represents a Transformer feed-forward network."""

    @abstractmethod
    def forward(self, seqs: Tensor) -> Tensor:
        """
        :param seqs:
            The sequences to project. *Shape:* :math:`(N,S,M)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`M`
            is the dimensionality of the model.

        :returns:
            The projected sequences. *Shape:* Same as ``seqs``.
        """

    if TYPE_CHECKING:
        __call__ = forward


@final
class StandardFeedForwardNetwork(FeedForwardNetwork):
    """Represents a Transformer feed-forward network as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        bias: bool,
        *,
        inner_activation: Module | None = None,
        inner_dropout_p: float = 0.0,
        proj_init_fn: Callable[[Linear], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param inner_dim:
            The dimensionality of the inner projection layer.
        :param bias:
            If ``True``, both the inner and output projection learn an additive
            bias.
        :param inner_activation:
            The activation to apply to outputs of the inner projection layer. If
            ``None``, :func:`~torch.nn.ReLU` will be used.
        :param inner_dropout_p:
            The dropout probability on outputs of the inner projection layer.
        :param proj_init_fn:
            The callable to initialize the inner and output projections.
        """
        super().__init__()

        self.inner_proj = Linear(
            model_dim, inner_dim, bias, init_fn=proj_init_fn, device=device, dtype=dtype
        )

        if inner_activation is not None:
            self.inner_activation = inner_activation
        else:
            self.inner_activation = ReLU()

        if inner_dropout_p > 0.0:
            inner_dropout = Dropout(inner_dropout_p)
        else:
            inner_dropout = None

        self.inner_dropout: Dropout | None

        self.register_module("inner_dropout", inner_dropout)

        self.output_proj = Linear(
            inner_dim, model_dim, bias, init_fn=proj_init_fn, device=device, dtype=dtype
        )

    @override
    def forward(self, seqs: Tensor) -> Tensor:
        seqs = self.inner_proj(seqs)

        seqs = self.inner_activation(seqs)

        if self.inner_dropout is not None:
            seqs = self.inner_dropout(seqs)

        seqs = self.output_proj(seqs)

        return seqs


@final
class DauphinFeedForwardNetwork(FeedForwardNetwork):
    """Represents a GLU-based Transformer feed-forward network as described in
    :cite:t:`https://doi.org/10.48550/arXiv.1612.08083`"""

    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        bias: bool,
        *,
        inner_activation: Module | None = None,
        inner_dropout_p: float = 0.0,
        proj_init_fn: Callable[[Linear], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param inner_dim:
            The dimensionality of the inner projection layer.
        :param bias:
            If ``True``, both the inner and output projection learn an additive
            bias.
        :param inner_activation:
            The activation to apply to outputs of the inner projection layer. If
            ``None``, :func:`~torch.nn.Sigmoid` will be used.
        :param inner_dropout_p:
            The dropout probability on outputs of the inner projection layer.
        :param proj_init_fn:
            The callable to initialize the inner and output projections.
        """
        super().__init__()

        self.inner_proj = Linear(
            model_dim,
            inner_dim * 2,
            bias,
            init_fn=proj_init_fn,
            device=device,
            dtype=dtype,
        )

        if inner_activation is not None:
            self.inner_activation = inner_activation
        else:
            self.inner_activation = Sigmoid()

        if inner_dropout_p > 0.0:
            inner_dropout = Dropout(inner_dropout_p)
        else:
            inner_dropout = None

        self.inner_dropout: Dropout | None

        self.register_module("inner_dropout", inner_dropout)

        self.output_proj = Linear(
            inner_dim, model_dim, bias, device=device, dtype=dtype
        )

    @override
    def forward(self, seqs: Tensor) -> Tensor:
        seqs = self.inner_proj(seqs)

        split1, split2 = seqs.chunk(2, dim=-1)

        seqs = self.inner_activation(split1) * split2  # gate

        if self.inner_dropout is not None:
            seqs = self.inner_dropout(seqs)

        seqs = self.output_proj(seqs)

        return seqs


@final
class GLUFeedForwardNetwork(FeedForwardNetwork):
    """Represents a GLU-based Transformer feed-forward network as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2002.05202`"""

    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        bias: bool,
        *,
        gate_activation: Module | None = None,
        inner_dim_scale: float = 2 / 3,
        inner_dim_to_multiple: int = 1,
        inner_dropout_p: float = 0.0,
        proj_init_fn: Callable[[Linear], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param inner_dim:
            The non-scaled dimensionality of the inner projection layer.
        :param bias:
            If ``True``, all projections learn an additive bias.
        :param gate_activation:
            The activation to apply to outputs of the gate projection. If
            ``None``, :func:`~torch.nn.SiLU` will be used.
        :param inner_dim_scale:
            The scale factor for the dimensionality of the inner projection
            layer.
        :param inner_dim_to_multiple:
            The dimensionality of the inner projection layer is rounded up to
            the nearest multiple of this value.
        :param inner_dropout_p:
            The dropout probability on outputs of the inner projection layer.
        :param proj_init_fn:
            The callable to initialize the inner, gate, and output projections.
        """
        super().__init__()

        self.inner_dim_scale = inner_dim_scale

        if inner_dim_scale != 1.0:
            inner_dim = int(inner_dim * inner_dim_scale)

        self.inner_dim_to_multiple = inner_dim_to_multiple

        if inner_dim_to_multiple != 1:
            inner_dim = inner_dim_to_multiple * (
                (inner_dim + inner_dim_to_multiple - 1) // inner_dim_to_multiple
            )

        self.gate_proj = Linear(
            model_dim,
            inner_dim,
            bias,
            init_fn=proj_init_fn,
            device=device,
            dtype=dtype,
        )

        if gate_activation is not None:
            self.gate_activation = gate_activation
        else:
            self.gate_activation = SiLU()

        self.inner_proj = Linear(
            model_dim,
            inner_dim,
            bias,
            init_fn=proj_init_fn,
            device=device,
            dtype=dtype,
        )

        if inner_dropout_p > 0.0:
            inner_dropout = Dropout(inner_dropout_p)
        else:
            inner_dropout = None

        self.inner_dropout: Dropout | None

        self.register_module("inner_dropout", inner_dropout)

        self.output_proj = Linear(
            inner_dim,
            model_dim,
            bias,
            init_fn=proj_init_fn,
            device=device,
            dtype=dtype,
        )

    @override
    def forward(self, seqs: Tensor) -> Tensor:
        seqs = self.forward_gateinner(seqs)

        seqs = self.forward_output(seqs)

        return seqs

    def forward_gateinner(self, seqs: Tensor) -> Tensor:
        """
        First step of the forward pass.
        Useful when interleaving computation and communication in EP/TP.
        """
        gate = self.gate_proj(seqs)

        gate = self.gate_activation(gate)

        seqs = self.inner_proj(seqs)

        seqs = seqs * gate

        del gate

        if self.inner_dropout is not None:
            seqs = self.inner_dropout(seqs)

        return seqs

    def forward_output(self, seqs: Tensor) -> Tensor:
        """
        Second step of the forward pass.
        Useful when interleaving computation and communication in EP/TP.
        """
        seqs = self.output_proj(seqs)

        return seqs

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"inner_dim_scale={self.inner_dim_scale:G}, "
            f"inner_dim_to_multiple={self.inner_dim_to_multiple}"
        )
