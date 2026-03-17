# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, final

import torch
import torch.nn.functional as F
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

        self.inner_proj = ColumnShardedLinear(
            model_dim,
            inner_dim,
            bias,
            gather_output=False,
            init_fn=proj_init_fn,
            gangs=gangs,
            device=device,
            dtype=dtype,
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

        self.output_proj = RowShardedLinear(
            inner_dim,
            model_dim,
            bias,
            scatter_input=False,
            init_fn=proj_init_fn,
            gangs=gangs,
            device=device,
            dtype=dtype,
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

        self.inner_proj = ColumnShardedLinear(
            model_dim,
            inner_dim * 2,
            bias,
            gather_output=False,
            init_fn=proj_init_fn,
            gangs=gangs,
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

        self.output_proj = RowShardedLinear(
            inner_dim,
            model_dim,
            bias,
            scatter_input=False,
            init_fn=proj_init_fn,
            gangs=gangs,
            device=device,
            dtype=dtype,
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
    """
    Represents a GLU-based Transformer feed-forward network as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2002.05202`
    """

    gate_proj: Projection
    inner_proj: Projection
    output_proj: Projection
    gate_activation: Module
    inner_dropout: Dropout | None
    inner_dim_scale: float
    inner_dim_to_multiple: int
    activation_sparsity: float

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
        activation_sparsity: float = 0.0,
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

        ``activation_sparsity`` specifies the target sparsity ratio for Gaussian
        top-k sparsification applied to gate outputs before activation. A value of
        0.95 means 95% of activations are zeroed. Default is 0.0 (no sparsification).

        If ``proj_init_fn`` is provided, it will be used to initialize the inner
        and output projections in :meth:`reset_parameters`.

        If ``gangs`` is provided, it will be used to shard the module for tensor
        parallelism.
        """
        super().__init__()

        self.inner_dim_scale = inner_dim_scale
        self.inner_dim_to_multiple = inner_dim_to_multiple
        self.activation_sparsity = activation_sparsity

        if inner_dim_scale != 1.0:
            inner_dim = int(inner_dim * inner_dim_scale)

        if inner_dim_to_multiple != 1:
            inner_dim = inner_dim_to_multiple * (
                (inner_dim + inner_dim_to_multiple - 1) // inner_dim_to_multiple
            )

        self.gate_proj = ColumnShardedLinear(
            model_dim,
            inner_dim,
            bias,
            gather_output=False,
            init_fn=proj_init_fn,
            gangs=gangs,
            device=device,
            dtype=dtype,
        )

        if gate_activation is not None:
            self.gate_activation = gate_activation
        else:
            self.gate_activation = SiLU()

        self.inner_proj = ColumnShardedLinear(
            model_dim,
            inner_dim,
            bias,
            gather_output=False,
            init_fn=proj_init_fn,
            gangs=gangs,
            device=device,
            dtype=dtype,
        )

        if inner_dropout_p > 0.0:
            inner_dropout = Dropout(inner_dropout_p)
        else:
            inner_dropout = None

        self.inner_dropout: Dropout | None

        self.register_module("inner_dropout", inner_dropout)

        self.output_proj = RowShardedLinear(
            inner_dim,
            model_dim,
            bias,
            scatter_input=False,
            init_fn=proj_init_fn,
            gangs=gangs,
            device=device,
            dtype=dtype,
        )

    def _gaussian_topk(self, inputs: Tensor) -> Tensor:
        """Apply Gaussian top-k sparsification to inputs.

        Computes a cutoff threshold based on the mean and standard deviation
        of the input values, then zeros out values below the threshold.

        :param inputs: Input tensor to sparsify.
        :returns: Sparsified tensor with values below threshold set to zero.
        """
        target_sparsity = torch.tensor(
            self.activation_sparsity, dtype=torch.float32, device=inputs.device
        )

        # Compute threshold multiplier using inverse CDF of standard normal
        normal_dist = torch.distributions.normal.Normal(0, 1)
        std_multiplier = normal_dist.icdf(target_sparsity).to(inputs.dtype)

        # Compute per-sequence statistics
        inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
        inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)

        # Apply threshold: keep only values > mean + std*multiplier
        cutoff = inputs_mean + inputs_std * std_multiplier
        return F.relu(inputs - cutoff)

    @override
    def forward(self, seqs: Tensor) -> Tensor:
        gate = self.gate_proj(seqs)

        # Apply Gaussian top-k sparsification if enabled (before activation)
        if self.activation_sparsity > 0.0:
            gate = self._gaussian_topk(gate)

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


@final
class AltUpFeedForwardNetwork(FeedForwardNetwork):
    """GLU-based FFN with GELU activation for Gemma3n local layers.

    Uses alternating up-projection pattern with GELU gating instead of SiLU.
    Uses tanh-approximated GELU to match HuggingFace implementation.
    """

    gate_proj: Projection
    inner_proj: Projection
    output_proj: Projection
    activation_sparsity: float

    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        bias: bool,
        *,
        activation_sparsity: float = 0.0,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        ``model_dim`` and ``inner_dim`` specify the dimensionality of the model
        and the inner projection respectively.

        If ``bias`` is ``True``, the gate, inner, and output projections will
        learn an additive bias.

        The gate activation uses tanh-approximated GELU to match HuggingFace.

        ``activation_sparsity`` specifies the target sparsity ratio for Gaussian
        top-k sparsification applied to gate activations before GELU. A value of
        0.95 means 95% of activations are zeroed. Default is 0.0 (no sparsification).
        """
        super().__init__()

        self.gate_proj = Linear(model_dim, inner_dim, bias, device=device, dtype=dtype)
        self.inner_proj = Linear(model_dim, inner_dim, bias, device=device, dtype=dtype)
        self.output_proj = Linear(
            inner_dim, model_dim, bias, device=device, dtype=dtype
        )
        self.activation_sparsity = activation_sparsity

    def _gaussian_topk(self, inputs: Tensor) -> Tensor:
        """Apply Gaussian top-k sparsification to inputs.

        Computes a cutoff threshold based on the mean and standard deviation
        of the input values, then zeros out values below the threshold.

        :param inputs: Input tensor to sparsify.
        :returns: Sparsified tensor with values below threshold set to zero.
        """
        import torch

        target_sparsity = torch.tensor(
            self.activation_sparsity, dtype=torch.float32, device=inputs.device
        )

        # Compute threshold multiplier using inverse CDF of standard normal
        normal_dist = torch.distributions.normal.Normal(0, 1)
        std_multiplier = normal_dist.icdf(target_sparsity).to(inputs.dtype)

        # Compute per-sequence statistics
        inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
        inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)

        # Apply threshold: keep only values > mean + std*multiplier
        cutoff = inputs_mean + inputs_std * std_multiplier
        return F.relu(inputs - cutoff)

    @override
    def forward(self, seqs: Tensor) -> Tensor:
        gate = self.gate_proj(seqs)

        # Apply Gaussian top-k sparsification if enabled
        if self.activation_sparsity > 0.0:
            gate = self._gaussian_topk(gate)

        # Use tanh-approximated GELU to match HuggingFace
        gate = F.gelu(gate, approximate="tanh")

        seqs = self.inner_proj(seqs)
        seqs = seqs * gate

        del gate

        seqs = self.output_proj(seqs)

        return seqs

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return ""
