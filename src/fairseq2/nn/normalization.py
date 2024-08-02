# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal, Optional, Union, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import layer_norm
from typing_extensions import override

try:
    from apex.normalization.fused_layer_norm import (  # type: ignore[import]
        fused_rms_norm,
        fused_rms_norm_affine,
    )

    _has_apex = True
except ImportError:
    _has_apex = False

from fairseq2.typing import DataType, Device


class LayerNorm(Module, ABC):
    """Applies Layer Normalization to incoming data."""

    normalized_shape: tuple[int, ...]
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
            f"eps={self.eps:G}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


@final
class StandardLayerNorm(LayerNorm):
    """Applies Layer Normalization to incoming data as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1607.06450`."""

    @override
    def forward(self, x: Tensor) -> Tensor:
        return layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


@final
class RMSNorm(LayerNorm):
    """Applies Root Mean Square Layer Normalization to incoming data as
    described in :cite:t:`https://doi.org/10.48550/arxiv.1910.07467`."""

    _impl: str
    _use_apex: bool

    def __init__(
        self, *args: Any, impl: Literal["auto", "py", "apex"] = "auto", **kwargs: Any
    ) -> None:
        """
        See :class:`LayerNorm` for ``args`` and ``kwargs``.

        :param impl:
            The underlying implementation. If 'auto', attempts to use APEX if
            installed; otherwise, falls back to Python.
        """
        super().__init__(*args, **kwargs)

        if impl == "auto":
            impl = "apex" if _has_apex and self.bias is None else "py"

        if impl == "apex":
            if not _has_apex:
                raise RuntimeError(
                    "`impl` is 'apex', but no APEX installation can be found."
                )

            if self.bias is not None:
                raise RuntimeError(
                    "`impl` is 'apex', but APEX does not support the `bias` parameter."
                )

        self._impl = impl

        self._use_apex = impl == "apex"

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self._use_apex and x.is_cuda:
            return self._apex_forward(x)

        # For numerical stability normalize in single precision.
        x = self._normalize(x.float()).type_as(x)

        if self.weight is not None:
            x = x * self.weight

            if self.bias is not None:
                x = x + self.bias

        return x

    def _apex_forward(self, x: Tensor) -> Tensor:
        if self.weight is None:
            return fused_rms_norm(x, self.normalized_shape, self.eps)  # type: ignore[no-any-return]

        return fused_rms_norm_affine(x, self.weight, self.normalized_shape, self.eps)  # type: ignore[no-any-return]

    def _normalize(self, x: Tensor) -> Tensor:
        dims = [-i for i in range(len(self.normalized_shape), 0, -1)]

        # Unlike the reference implementation, we add the epsilon before square
        # root similar to LLaMA. APEX does the same.
        return x * torch.rsqrt(x.pow(2).mean(dims, keepdim=True) + self.eps)

    @property
    def impl(self) -> str:
        """The underlying implementation."""
        return self._impl

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        if self._impl != "py":
            s = f"{s}, impl={self._impl}"

        return s
