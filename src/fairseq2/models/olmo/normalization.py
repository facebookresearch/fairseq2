# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OLMO-specific normalization layers."""

from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from typing_extensions import final, override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.nn import LayerNorm


@final
class OLMORMSNorm(LayerNorm):
    """OLMO Root Mean Square Layer Normalization.

    The mathematical representation of this RMSNorm is identical to the LLaMA implementation.

    The key difference from standard RMSNorm is the order of operations:
    - Standard: normalize -> cast to original dtype -> multiply by weight
    - OLMO:     normalize -> multiply by weight     -> cast to original dtype

    This matches the [HuggingFace OLMO implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmo/modular_olmo.py) where the weight
    and normalized hidden states are multiplied before converting back
    to the input dtype.
    """

    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        bias: bool,
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        init_fn: Callable[[OLMORMSNorm], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
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
        """Reset the parameters and buffers of the module."""
        if self.init_fn is not None:
            self.init_fn(self)
        else:
            if self.weight is not None:
                nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    @override
    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype

        # For numerical stability, normalize in single precision
        x = x.float()

        # Normalize
        x = self._normalize(x)

        # Apply weight BEFORE casting back (OLMO-specific)
        if self.weight is not None:
            x = x * self.weight

            if self.bias is not None:
                x = x + self.bias

        # Cast back to original dtype
        return x.to(input_dtype)

    def _normalize(self, x: Tensor) -> Tensor:
        """RMS normalization."""
        dims = [-i for i in range(len(self.normalized_shape), 0, -1)]
        return x * torch.rsqrt(x.pow(2).mean(dims, keepdim=True) + self.eps)

    @override
    def extra_repr(self) -> str:
        s = (
            f"normalized_shape={self.normalized_shape}, "
            f"eps={self.eps:G}, "
            f"elementwise_affine={self.elementwise_affine}"
        )
        return s
