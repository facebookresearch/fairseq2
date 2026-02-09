# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device


class ResidualConnect(Module, ABC):
    """Represents a residual connection."""

    @abstractmethod
    def forward(self, seqs: Tensor, residual: Tensor) -> Tensor:
        """
        :param seqs: The sequences output by a module. *Shape:* :math:`(N,S,M)`,
            where :math:`N` is the batch size, :math:`S` is the sequence length,
            and :math:`M` is the dimensionality of the model.
        :param residual: The input sequences to the module. *Shape:* Same as
            ``seqs``.

        :returns: The output sequences with residuals applied. *Shape:* Same as
            ``seqs``.
        """

    if TYPE_CHECKING:
        __call__ = forward


@final
class AdditiveResidualConnect(ResidualConnect):
    """Sums inputs and outputs of a module."""

    @override
    def forward(self, seqs: Tensor, residual: Tensor) -> Tensor:
        return seqs + residual


@final
class ScaledResidualConnect(ResidualConnect):
    """
    Scales residuals by a constant factor before adding them to the output of a
    Transformer module.
    """

    def __init__(self, scale: float) -> None:
        """
        :param scale: The scale factor.
        """
        self.scale = scale

    @override
    def forward(self, seqs: Tensor, residual: Tensor) -> Tensor:
        residual = self.scale * residual

        return seqs + residual

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"scale={self.scale}"


@final
class DropPathResidualConnect(ResidualConnect):
    """
    Drops entire sequences from module outputs before adding residuals which
    effectively results in stochastic depth as described in section 3 of
    :cite:t:`https://doi.org/10.48550/arxiv.1603.09382`.

    .. note::
        This implementation is mostly adapted from Ross Wightman's ``drop_path``
        function in timm.
    """

    def __init__(self, drop_p: float, scale_by_keep: bool = True) -> None:
        """
        :param drop_p: The probability of dropping sequences from module outputs.
        :param scale_by_keep: If ``True``, non-dropped sequences will be scaled
            by the keep probability (i.e. ``1 - drop_p``) as in EfficientNet.
        """
        super().__init__()

        self.drop_p = drop_p
        self.scale_by_keep = scale_by_keep

    @override
    def forward(self, seqs: Tensor, residual: Tensor) -> Tensor:
        if not self.training or self.drop_p == 0.0:
            return seqs + residual

        shape = [seqs.size(0)] + [1] * (seqs.ndim - 1)

        keep_p = 1.0 - self.drop_p

        # (N)
        drop_mask = torch.rand(shape, device=seqs.device, dtype=seqs.dtype) + keep_p

        drop_mask.floor_()  # binarize

        if self.scale_by_keep:
            seqs = seqs / keep_p

        return (seqs * drop_mask) + residual

    @override
    def extra_repr(self) -> str:
        return f"drop_p={self.drop_p}, scale_by_keep={self.scale_by_keep}"


@final
class LAuReLResidualConnect(ResidualConnect):
    """
    Learned Augmented Residual Layer (LAuReL) residual connection.

    Replaces identity residual with low-rank learned transformation followed by
    normalization. Used in Gemma3n to enhance residual paths with minimal
    parameter overhead.

    Reference: https://arxiv.org/abs/2411.07501
    """

    def __init__(
        self,
        model_dim: int,
        rank: int,
        layer_norm: Module,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param model_dim: Dimensionality of the model.
        :param rank: Rank of the low-rank factorization.
        :param layer_norm: Layer normalization applied after residual connection.
        """
        super().__init__()

        from fairseq2.nn.projection import Linear

        self.linear_left = Linear(
            model_dim, rank, bias=False, device=device, dtype=dtype
        )
        self.linear_right = Linear(
            rank, model_dim, bias=False, device=device, dtype=dtype
        )
        self.layer_norm = layer_norm

    @override
    def forward(self, seqs: Tensor, residual: Tensor) -> Tensor:
        residual_transformed = self.linear_right(self.linear_left(residual))
        return self.layer_norm(seqs + residual_transformed)

    @override
    def extra_repr(self) -> str:
        return f"rank={self.linear_left.out_features}"

