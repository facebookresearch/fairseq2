# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor


def repeat_interleave(x: Tensor, dim: int, repeat: int) -> Tensor:
    """Repeat elements of a tensor.

    :param x:
        The input tensor.
    :param dim:
        The dimension along which to repeat values.
    :param repeat:
        The number of repetitions.

    :returns:
        The repeated tensor which has the same shape as input, except along the
        given axis.

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
