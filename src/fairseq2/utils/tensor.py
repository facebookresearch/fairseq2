# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import torch
from torch import Tensor

from fairseq2.data_type import DataType
from fairseq2.device import CPU, Device

TensorData: TypeAlias = int | float | Sequence[int] | Sequence[float]


def to_tensor(
    data: TensorData, device: Device | None = None, dtype: DataType | None = None
) -> Tensor:
    if device is None or device.type != "cuda":
        return torch.tensor(data, dtype=dtype, device=device)

    t = torch.tensor(data, device=CPU, pin_memory=True)

    return t.to(device, non_blocking=True)


def unsqueeze(x: Tensor, dim: int, count: int = 1) -> Tensor:
    for _ in range(count):
        x = x.unsqueeze(dim=dim)

    return x


def repeat_interleave(x: Tensor, dim: int, repeat: int) -> Tensor:
    """
    Repeats elements of a tensor.

    :param x: The input tensor.
    :param dim: The dimension along which to repeat values.
    :param repeat: The number of repetitions.

    :returns: The repeated tensor which has the same shape as input, except
        along the given axis.

    .. note::
        This is a lightweight version of :func:`torch.repeat_interleave` that
        is faster for repetitions along a single dimension.
    """
    if repeat == 1:
        return x

    shape = [-1] * (x.ndim + 1)

    if dim < 0:
        dim += x.ndim

    shape[dim + 1] = repeat

    return x.unsqueeze(dim + 1).expand(shape).flatten(dim, dim + 1)
