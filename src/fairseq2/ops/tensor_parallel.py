# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.autograd import Function

from fairseq2.gang import Gang, ReduceOperation

# mypy: disable-error-code="no-any-return,override"


def reduce(x: Tensor, gang: Gang) -> Tensor:
    """Reduce ``x`` across all processes in ``gang``.

    This is an autograd-aware operation and the backward pass will return the
    all-reduced gradient of ``x``.
    """
    return _ReduceFunction.apply(x, gang)


class _ReduceFunction(Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, gang: Gang) -> Tensor:
        x = x.detach().clone()

        gang.all_reduce(x, ReduceOperation.SUM)

        return x

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None]:
        return grad_output, None, None


def scatter(x: Tensor, gang: Gang, dim: int = -1) -> Tensor:
    """Scatter ``x`` across all processes in ``gang`` over ``dim``.

    This is an autograd-aware operation and the backward pass will return the
    all-gathered gradient of ``x``.
    """
    return _ScatterFunction.apply(x, gang, dim)


class _ScatterFunction(Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, gang: Gang, dim: int) -> Tensor:
        ctx.dim = dim
        ctx.gang = gang

        return _do_scatter(x, gang, dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None]:
        x = _do_gather(grad_output, ctx.gang, ctx.dim)

        return x, None, None


def gather(x: Tensor, gang: Gang, dim: int = -1) -> Tensor:
    """Gather ``x`` across all processes in ``gang`` over ``dim``.

    This is an autograd-aware operation and the backward pass will return the
    scattered gradient of ``x``.
    """
    return _GatherFunction.apply(x, gang, dim)


class _GatherFunction(Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, gang: Gang, dim: int) -> Tensor:
        ctx.dim = dim
        ctx.gang = gang

        return _do_gather(x, gang, dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None]:
        x = _do_scatter(grad_output, ctx.gang, ctx.dim)

        return x, None, None


def reduce_on_backward(x: Tensor, gang: Gang) -> Tensor:
    """Reduce the gradient of ``x`` across all processes in ``gang``."""
    return _ReduceOnBackwardFunction.apply(x, gang)


class _ReduceOnBackwardFunction(Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, gang: Gang) -> Tensor:
        ctx.gang = gang

        return x

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        ctx.gang.all_reduce(grad_output, ReduceOperation.SUM)

        return grad_output, None


def _do_scatter(x: Tensor, gang: Gang, dim: int) -> Tensor:
    if gang.size == 1:
        return x

    dim_size = x.size(dim)

    if dim_size % gang.size != 0:
        raise ValueError(
            f"Size of dimension {dim} of `x` must be a multiple of `gang.size` ({gang.size}), but is {dim_size} instead."
        )

    splits = x.split(dim_size // gang.size, dim=dim)

    return splits[gang.rank].contiguous()


def _do_gather(x: Tensor, gang: Gang, dim: int) -> Tensor:
    if gang.size == 1:
        return x

    splits = [torch.empty_like(x) for _ in range(gang.size)]

    gang.all_gather_to_list(splits, x)

    return torch.cat(splits, dim=dim)
