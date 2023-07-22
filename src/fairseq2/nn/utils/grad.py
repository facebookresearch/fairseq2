# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Tuple

from torch import Tensor
from torch.autograd import Function


def scale_grad(x: Tensor, scale: float) -> Tensor:
    """Scale the gradient of ``x`` during backpropagation.

    This might be used to, for example, allow one part of a model to learn at a
    lower rate than the rest.

    :param x:
        The input tensor.
    :param scale:
        The scale factor of the gradient.
    """
    return _GradScaler.apply(x, scale)  # type: ignore[no-any-return]


class _GradScaler(Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, scale: float) -> Tensor:  # type: ignore[override]
        if not x.dtype.is_floating_point:
            raise TypeError(
                f"`x` must be a float tensor, but is of type `{x.dtype}` instead."
            )

        ctx.scale = scale

        return x.clone().detach().requires_grad_(True)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None]:  # type: ignore[override]
        return grad_output * ctx.scale, None
