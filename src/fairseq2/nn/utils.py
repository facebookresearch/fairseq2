# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from torch import Tensor

neg_inf = float("-inf")


def to_float_mask(mask: Tensor, dtype: Any = torch.float32) -> Tensor:
    """Converts a boolean mask to its floating-point equivalent.

    If ``mask`` is of type ``torch.bool``, all its ``False`` values will be
    converted to zero and all its ``True`` values will be converted to negative
    infinity (e.g.  ``float("-inf")``); otherwise, it will be returned as is
    without any conversion.

    :param mask:
        The mask tensor. *Shape:* Any.
    :param dtype:
        The floating-point type of the converted mask.

    :returns:
        The floating-point equivalent of ``mask`` if ``mask`` is of type
        ``torch.bool``; otherwise, ``mask`` itself.
    """
    if mask.dtype == torch.bool:
        if not dtype.is_floating_point:
            raise ValueError("`dtype` must be a floating-point type.")

        float_mask = mask.new_zeros(mask.shape, dtype=dtype)

        return float_mask.masked_fill_(mask, neg_inf)
    else:
        return mask
