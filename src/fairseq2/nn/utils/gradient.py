# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from torch.autograd import Function
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_

from fairseq2.gang import Gang
from fairseq2.utils.logging import get_log_writer

log = get_log_writer(__name__)


def scale_gradients(module: Module, value: float) -> None:
    """Scale gradients of ``module`` by ``value``.

    :param module:
        The module whose gradients to clip.
    :param value:
        The value to scale by.
    """
    for param in module.parameters():
        if param.grad is not None:
            param.grad *= value


def scale_gradient(x: Tensor, scale: float) -> Tensor:
    """Scale the gradient of ``x`` during backpropagation.

    This is typically used to allow one part of a model to learn at a lower rate
    than the rest.

    :param x:
        The input tensor.
    :param scale:
        The scale factor of the gradient.
    """
    return _GradientScaler.apply(x, scale)  # type: ignore[no-any-return]


class _GradientScaler(Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, scale: float) -> Tensor:  # type: ignore[override]
        if not x.dtype.is_floating_point:
            raise TypeError(
                f"`x` must be a float tensor, but is of type `{x.dtype}` instead."
            )

        ctx.scale = scale

        return x.clone().detach().requires_grad_(True)

    @staticmethod
    def backward(ctx: Any, output: Tensor) -> Tuple[Tensor, None]:  # type: ignore[override]
        return output * ctx.scale, None


def clip_gradient_norm(
    module: Module, max_norm: Optional[float], norm_type: float = 2.0
) -> Tensor:
    """Clip the gradient norms ``module``.

    :param module:
        The module whose gradients to clip.
    :param max_norm:
        The maximum norm.
    :param norm_type:
        The type of the used p-norm.
    """
    if max_norm is None:
        max_norm = torch.inf

    if isinstance(module, FSDP):
        if not module.check_is_root():
            raise ValueError("`module` must be the root FSDP module.")

        return module.clip_grad_norm_(max_norm, norm_type)

    return clip_grad_norm_(
        module.parameters(), max_norm, norm_type, error_if_nonfinite=False
    )


def check_gradient_norms(local_norm: Tensor, gang: Gang, step_nr: int) -> bool:
    """Sanity check the total gradient norm across all processes.

    :param local_norm:
        The local total gradient norm.
    :param gang:
        The gang over which to check the total gradient norm.
    :param step_nr:
        The number of the training step. Used for logging purposes.
    """
    if gang.size == 1:
        return True

    norms = torch.zeros((gang.size,), device=gang.device, dtype=local_norm.dtype)

    gang.all_gather(norms, local_norm)

    if all_finite := norms.isfinite().all():
        delta = (norms - norms[0]).abs().max() / (norms[0] + 1e-6)

        if (delta < 1e-6).all():
            return True
    else:
        if all_finite.logical_not().all():  # Check if all Inf/NaN.
            return True

    if log.is_enabled_for(logging.ERROR):
        s = "\n".join(f"Rank {r:3d} = {g:.8f}" for r, g in enumerate(norms.tolist()))

        log.error("Gradients are inconsistent between processes at step {}. Gradient Norms:\n{}", step_nr, s)  # fmt: skip

    return False
