# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor


def to_float_mask(mask: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    """Convert a boolean mask to its floating-point equivalent.

    If ``mask`` is of type ``torch.bool``, all its ``False`` values will be
    converted to zero and all its ``True`` values will be converted to negative
    infinity (e.g. ``float("-inf")``); otherwise, it will be returned as is
    without any conversion.

    :param mask:
        The mask tensor. *Shape:* Any.
    :param dtype:
        The floating-point type of the converted mask.

    :returns:
        The floating-point equivalent of ``mask`` if ``mask`` is of type
        ``torch.bool``; otherwise, ``mask`` itself.
    """
    if mask is not None and mask.dtype == torch.bool:
        if not dtype.is_floating_point:
            raise ValueError(
                f"`dtype` must be a floating-point type, but is `{dtype}` instead."
            )

        return mask.new_zeros(mask.shape, dtype=dtype).masked_fill_(mask, _neg_inf)
    else:
        return mask


_neg_inf = float("-inf")
