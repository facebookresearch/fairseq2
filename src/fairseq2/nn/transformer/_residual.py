# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter
from typing_extensions import override

from fairseq2.typing import DataType, Device


class ResidualConnect(Module, ABC):
    """Represents a residual connection within a Transformer layer."""

    @abstractmethod
    def forward(self, seqs: Tensor, residual: Tensor) -> Tensor:
        """
        :param seqs: The sequences output by a module such as a multi-head
            attention layer or a feed-forward network. *Shape:* :math:`(N,S,M)`,
            where :math:`N` is the batch size, :math:`S` is the sequence length,
            and :math:`M` is the dimensionality of the model.
        :param residual: The input sequences to the module. *Shape:* Same as
            ``seqs``.

        :returns: The output sequences with residuals applied. *Shape:* Same as
            ``seqs``.
        """


@final
class StandardResidualConnect(ResidualConnect):
    """Sums inputs and outputs of a Transformer module."""

    @override
    def forward(self, seqs: Tensor, residual: Tensor) -> Tensor:
        return seqs + residual


@final
class ScaledResidualConnect(ResidualConnect):
    """
    Scales residuals by a constant factor before adding them to the output of a
    Transformer module.
    """

    scale: float

    def __init__(self, scale: float) -> None:
        """
        :param scale: The scale factor.
        """
        self.scale = scale

    @override
    def forward(self, seqs: Tensor, residual: Tensor) -> Tensor:
        residual = self.scale * residual

        return seqs + residual


@final
class NormFormerResidualConnect(ResidualConnect):
    """
    Scales residuals by a learned factor before adding them to the output of a
    feed-forward network as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2110.09456`.
    """

    scale_proj: Parameter

    def __init__(
        self,
        model_dim: int,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param model_dim: The dimensionality of the model.
        """
        super().__init__()

        self.scale_proj = Parameter(
            torch.empty((model_dim,), device=device, dtype=dtype)
        )

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        nn.init.ones_(self.scale_proj)

    @override
    def forward(self, seqs: Tensor, residual: Tensor) -> Tensor:
        residual = self.scale_proj * residual

        return seqs + residual


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

    drop_p: float
    scale_by_keep: bool

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

    def extra_repr(self) -> str:
        return f"drop_p={self.drop_p}, scale_by_keep={self.scale_by_keep}"
