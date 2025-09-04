# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, final

import torch
import torch.nn as nn
from torch import Size, Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import layer_norm
from typing_extensions import override

try:
    from torch.nn.functional import rms_norm  # type: ignore[import]
except ImportError:
    _has_rms_norm = False
else:
    _has_rms_norm = True

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.nn.utils.module import get_name_or_self


class LayerNorm(Module, ABC):
    """Applies Layer Normalization to incoming data."""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: The input to normalize. *Shape:* :math:`(*,H)`, where :math:`H`
            is :attr:`normalized_shape`.

        :returns: The normalized output. *Shape:* Same as ``x``.
        """

    if TYPE_CHECKING:
        __call__ = forward


@final
class StandardLayerNorm(LayerNorm):
    """
    Applies Layer Normalization to incoming data as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1607.06450`.
    """

    def __init__(
        self,
        normalized_shape: int | Sequence[int] | Size,
        bias: bool,
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        cast_fp32: bool = False,
        init_fn: Callable[[StandardLayerNorm], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param normalized_shape: The shape over which to normalize incoming data.
            For example, if the shape is ``(3, 5)``, the incoming data is
            normalized over the last 2 dimensions (i.e. ``input.mean((-2, -1))``).
        :param bias: If ``True``, learns an additive bias. Ignored if
            ``elementwise_affine`` is ``False``.
        :param eps: The value to add to the denominator for numerical stability.
        :param elementwise_affine: If ``True``, learns an affine transformation.
        """
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)

        self.eps = eps

        self.elementwise_affine = elementwise_affine

        self.cast_fp32 = cast_fp32

        if elementwise_affine:
            weight = Parameter(
                torch.empty(normalized_shape, device=device, dtype=dtype)
            )
        else:
            weight = None

        self.weight: Parameter | None

        self.register_parameter("weight", weight)

        if elementwise_affine and bias:
            bias_ = Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        else:
            bias_ = None

        self.bias: Parameter | None

        self.register_parameter("bias", bias_)

        self.init_fn = init_fn

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_fn is not None:
            self.init_fn(self)
        else:
            if self.weight is not None:
                nn.init.ones_(self.weight)

            if self.bias is not None:
                nn.init.zeros_(self.bias)

    @override
    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype

        w: Tensor | None
        b: Tensor | None

        w, b = self.weight, self.bias

        if self.cast_fp32:
            x = x.float()

            w = w.float() if w is not None else None
            b = b.float() if b is not None else None

        x = layer_norm(x, self.normalized_shape, w, b, self.eps)

        if self.cast_fp32:
            x = x.type(dtype)

        return x

    @override
    def extra_repr(self) -> str:
        s = (
            f"normalized_shape={self.normalized_shape}, "
            f"eps={self.eps:G}, "
            f"elementwise_affine={self.elementwise_affine}"
        )

        if self.init_fn is not None:
            init_fn = get_name_or_self(self.init_fn)

            s = f"{s}, init_fn={init_fn}"

        return s


@final
class RMSNorm(LayerNorm):
    """
    Applies Root Mean Square Layer Normalization to incoming data as described
    in :cite:t:`https://doi.org/10.48550/arxiv.1910.07467`.
    """

    def __init__(
        self,
        normalized_shape: int | Sequence[int] | Size,
        bias: bool,
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        init_fn: Callable[[RMSNorm], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param normalized_shape: The shape over which to normalize incoming data.
            For example, if the shape is ``(3, 5)``, the incoming data is
            normalized over the last 2 dimensions (i.e. ``input.mean((-2, -1))``).
        :param bias: If ``True``, learns an additive bias. Ignored if
            ``elementwise_affine`` is ``False``.
        :param eps: The value to add to the denominator for numerical stability.
        :param elementwise_affine: If ``True``, learns an affine transformation.
        """
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)

        self.eps = eps

        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            weight = Parameter(
                torch.empty(normalized_shape, device=device, dtype=dtype)
            )
        else:
            weight = None

        self.weight: Parameter | None

        self.register_parameter("weight", weight)

        if elementwise_affine and bias:
            bias_ = Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        else:
            bias_ = None

        self.bias: Parameter | None

        self.register_parameter("bias", bias_)

        self.init_fn = init_fn

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_fn is not None:
            self.init_fn(self)
        else:
            if self.weight is not None:
                nn.init.ones_(self.weight)

            if self.bias is not None:
                nn.init.zeros_(self.bias)

    @override
    def forward(self, x: Tensor) -> Tensor:
        if _has_rms_norm:
            return rms_norm(x, self.normalized_shape, self.weight, self.eps)

        # For numerical stability normalize in single precision.
        x = self._normalize(x.float()).type_as(x)

        if self.weight is not None:
            x = x * self.weight

            if self.bias is not None:
                x = x + self.bias

        return x

    def _normalize(self, x: Tensor) -> Tensor:
        dims = [-i for i in range(len(self.normalized_shape), 0, -1)]

        # Unlike the reference implementation, we add the epsilon before square
        # root similar to LLaMA.
        return x * torch.rsqrt(x.pow(2).mean(dims, keepdim=True) + self.eps)

    @override
    def extra_repr(self) -> str:
        s = (
            f"normalized_shape={self.normalized_shape}, "
            f"eps={self.eps:G}, "
            f"elementwise_affine={self.elementwise_affine}"
        )

        if self.init_fn is not None:
            init_fn = get_name_or_self(self.init_fn)

            s = f"{s}, init_fn={init_fn}"

        return s
