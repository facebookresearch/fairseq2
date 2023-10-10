# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Union, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import layer_norm

from fairseq2.typing import DataType, Device, finaloverride


class LayerNorm(Module, ABC):
    """Applies Layer Normalization to incoming data."""

    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool
    weight: Optional[Parameter]
    bias: Optional[Parameter]

    def __init__(
        self,
        normalized_shape: Union[int, Sequence[int], torch.Size],
        bias: bool,
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param normalized_shape:
            The shape over which to normalize incoming data. For example, if the
            shape is ``(3, 5)``, the incoming data is normalized over the last 2
            dimensions (i.e. ``input.mean((-2, -1))``).
        :param bias:
            If ``True``, learns an additive bias. Ignored if
            ``elementwise_affine`` is ``False``.
        :param eps:
            The value to add to the denominator for numerical stability.
        :param elementwise_affine:
            If ``True``, learns an affine transformation.
        """
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)

        self.eps = eps

        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = Parameter(
                torch.empty(normalized_shape, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("weight", None)

        if elementwise_affine and bias:
            self.bias = Parameter(
                torch.empty(normalized_shape, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        if self.weight is not None:
            nn.init.ones_(self.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        :param x:
            The input to normalize. *Shape:* :math:`(*,H)`, where :math:`H` is
            :attr:`normalized_shape`.

        :param:
            The normalized output. *Shape:* Same as ``x``.
        """

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, "
            f"eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


@final
class StandardLayerNorm(LayerNorm):
    """Applies Layer Normalization to incoming data as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1607.06450`."""

    @finaloverride
    def forward(self, x: Tensor) -> Tensor:
        return layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


@final
class RMSNorm(LayerNorm):
    """Applies Root Mean Square Layer Normalization to incoming data as
    described in :cite:t:`https://doi.org/10.48550/arxiv.1910.07467`."""

    @finaloverride
    def forward(self, x: Tensor) -> Tensor:
        # For numerical stability normalize in single precision.
        x = self._norm(x.float()).type_as(x)

        if self.weight is not None:
            x = x * self.weight

            if self.bias is not None:
                x = x + self.bias

        return x

    def _norm(self, x: Tensor) -> Tensor:
        dims = [-i for i in range(len(self.normalized_shape), 0, -1)]

        # Unlike the reference implementation, we add the epsilon before square
        # root similar to LLaMA.
        return x * torch.rsqrt(x.pow(2).mean(dims, keepdim=True) + self.eps)
