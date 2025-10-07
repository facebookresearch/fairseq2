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
from fairseq2.gang import Gangs
from fairseq2.nn import ColumnShardedLinear, Linear, Projection, RowShardedLinear


class FeedForwardNetwork(Module, ABC):
    """Represents a Transformer feed-forward network."""

    @abstractmethod
    def forward(self, seqs: Tensor) -> Tensor:
        """
        ``seqs`` is expected to be of shape :math:`(N,S,M)`, where :math:`N` is
        the batch size, :math:`S` is the sequence length, and :math:`M` is the
        dimensionality of the model.

        The processed sequences will have the same shape as ``seqs``.
        """

    if TYPE_CHECKING:
        __call__ = forward


@final
class StandardFeedForwardNetwork(FeedForwardNetwork):
    """
    Represents a Transformer feed-forward network as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`.
    """

    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        bias: bool,
        *,
        inner_activation: Module | None = None,
        inner_dropout_p: float = 0.0,
        proj_init_fn: Callable[[Linear], None] | None = None,
        gangs: Gangs | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        ``model_dim`` and ``inner_dim`` specify the dimensionality of the model
        and the inner projection respectively.

        If ``bias`` is ``True``, both the inner and output projections will
        learn an additive bias.

        If ``inner_activation`` is provided, it will be used as the activation
        to apply to the outputs of the inner projection; otherwise,
        :func:`~torch.nn.ReLU` will be used.

        ``inner_dropout_p`` specifies the dropout probability on the outputs of
        the inner projection.

        If ``proj_init_fn`` is provided, it will be used to initialize the inner
        and output projections in :meth:`reset_parameters`.

        If ``gangs`` is provided, it will be used to shard the module for tensor
        parallelism.
        """
        super().__init__()

        inner_proj = Linear(
            model_dim, inner_dim, bias, init_fn=proj_init_fn, device=device, dtype=dtype
        )

        self.inner_proj: Projection

        if gangs is None or gangs.tp.size == 1:
            self.inner_proj = inner_proj
        else:
            self.inner_proj = ColumnShardedLinear.from_linear(
                inner_proj, gangs.tp, gather_output=False
            )

            del inner_proj

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

        output_proj = Linear(
            inner_dim, model_dim, bias, init_fn=proj_init_fn, device=device, dtype=dtype
        )

        self.output_proj: Projection

        if gangs is None or gangs.tp.size == 1:
            self.output_proj = output_proj
        else:
            self.output_proj = RowShardedLinear.from_linear(
                output_proj, gangs.tp, scatter_input=False
            )

            del output_proj

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
    """
    Represents a GLU-based Transformer feed-forward network as described in
    :cite:t:`https://doi.org/10.48550/arXiv.1612.08083`
    """

    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        bias: bool,
        *,
        inner_activation: Module | None = None,
        inner_dropout_p: float = 0.0,
        proj_init_fn: Callable[[Linear], None] | None = None,
        gangs: Gangs | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        ``model_dim`` and ``inner_dim`` specify the dimensionality of the model
        and the inner projection respectively.

        If ``bias`` is ``True``, both the inner and output projections will
        learn an additive bias.

        If ``inner_activation`` is provided, it will be used as the activation
        to apply to the outputs of the inner projection; otherwise,
        :func:`~torch.nn.ReLU` will be used.

        ``inner_dropout_p`` specifies the dropout probability on the outputs of
        the inner projection.

        If ``proj_init_fn`` is provided, it will be used to initialize the inner
        and output projections in :meth:`reset_parameters`.

        If ``gangs`` is provided, it will be used to shard the module for tensor
        parallelism.
        """
        super().__init__()

        inner_proj = Linear(
            model_dim,
            inner_dim * 2,
            bias,
            init_fn=proj_init_fn,
            device=device,
            dtype=dtype,
        )

        self.inner_proj: Projection

        if gangs is None or gangs.tp.size == 1:
            self.inner_proj = inner_proj
        else:
            self.inner_proj = ColumnShardedLinear.from_linear(
                inner_proj, gangs.tp, gather_output=False
            )

            del inner_proj

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

        output_proj = Linear(inner_dim, model_dim, bias, device=device, dtype=dtype)

        self.output_proj: Projection

        if gangs is None or gangs.tp.size == 1:
            self.output_proj = output_proj
        else:
            self.output_proj = RowShardedLinear.from_linear(
                output_proj, gangs.tp, scatter_input=False
            )

            del output_proj

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
    """
    Represents a GLU-based Transformer feed-forward network as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2002.05202`
    """

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
        gangs: Gangs | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        ``model_dim`` and ``inner_dim`` specify the dimensionality of the model
        and the inner projection respectively.

        If ``bias`` is ``True``, both the inner and output projections will
        learn an additive bias.

        If ``gate_activation`` is provided, it will be used as the activation to
        apply to the outputs of the gate projection; otherwise,
        :func:`~torch.nn.SiLU` will be used.

        The dimensionality of the inner projection will be scaled by a factor
        of ``inner_dim_scale`` and be rounded up to the nearest multiple of
        ``inner_dim_to_multiple``.

        ``inner_dropout_p`` specifies the dropout probability on the outputs of
        the inner projection.

        If ``proj_init_fn`` is provided, it will be used to initialize the inner
        and output projections in :meth:`reset_parameters`.

        If ``gangs`` is provided, it will be used to shard the module for tensor
        parallelism.
        """
        super().__init__()

        self.inner_dim_scale = inner_dim_scale
        self.inner_dim_to_multiple = inner_dim_to_multiple

        if inner_dim_scale != 1.0:
            inner_dim = int(inner_dim * inner_dim_scale)

        if inner_dim_to_multiple != 1:
            inner_dim = inner_dim_to_multiple * (
                (inner_dim + inner_dim_to_multiple - 1) // inner_dim_to_multiple
            )

        gate_proj = Linear(
            model_dim,
            inner_dim,
            bias,
            init_fn=proj_init_fn,
            device=device,
            dtype=dtype,
        )

        self.gate_proj: Projection

        if gangs is None or gangs.tp.size == 1:
            self.gate_proj = gate_proj
        else:
            self.gate_proj = ColumnShardedLinear.from_linear(
                gate_proj, gangs.tp, gather_output=False
            )

            del gate_proj

        if gate_activation is not None:
            self.gate_activation = gate_activation
        else:
            self.gate_activation = SiLU()

        inner_proj = Linear(
            model_dim,
            inner_dim,
            bias,
            init_fn=proj_init_fn,
            device=device,
            dtype=dtype,
        )

        self.inner_proj: Projection

        if gangs is None or gangs.tp.size == 1:
            self.inner_proj = inner_proj
        else:
            self.inner_proj = ColumnShardedLinear.from_linear(
                inner_proj, gangs.tp, gather_output=False
            )

            del inner_proj

        if inner_dropout_p > 0.0:
            inner_dropout = Dropout(inner_dropout_p)
        else:
            inner_dropout = None

        self.inner_dropout: Dropout | None

        self.register_module("inner_dropout", inner_dropout)

        output_proj = Linear(
            inner_dim,
            model_dim,
            bias,
            init_fn=proj_init_fn,
            device=device,
            dtype=dtype,
        )

        self.output_proj: Projection

        if gangs is None or gangs.tp.size == 1:
            self.output_proj = output_proj
        else:
            self.output_proj = RowShardedLinear.from_linear(
                output_proj, gangs.tp, scatter_input=False
            )

            del output_proj

    @override
    def forward(self, seqs: Tensor) -> Tensor:
        gate = self.gate_proj(seqs)

        gate = self.gate_activation(gate)

        seqs = self.inner_proj(seqs)

        seqs = seqs * gate

        del gate

        if self.inner_dropout is not None:
            seqs = self.inner_dropout(seqs)

        seqs = self.output_proj(seqs)

        return seqs

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"inner_dim_scale={self.inner_dim_scale:G}, "
            f"inner_dim_to_multiple={self.inner_dim_to_multiple}"
        )
