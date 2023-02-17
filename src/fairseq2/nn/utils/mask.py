# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["to_float_mask", "to_padding_mask"]

import torch
from torch import Tensor

_neg_inf = float("-inf")


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
            raise ValueError("`dtype` must be a floating-point type.")

        float_mask = mask.new_zeros(mask.shape, dtype=dtype)

        return float_mask.masked_fill_(mask, _neg_inf)
    else:
        return mask


def to_padding_mask(seq_lens: Tensor, max_seq_len: int) -> Tensor:
    """Convert a sequence length array into a boolean padding mask.

    :param seq_lens:
        The sequence length array. Each entry defines the length of the sequence
        at the same index in the corresponding mini-batch. *Shape:* :math:`(N)`,
        :math:`(N,1)`, or :math:`()` when unbatched, where :math:`N` is the
        batch size.

    :returns:
        The padding mask. *Shape:* :math:`(N,S)`, or :math:`(S)` when unbatched,
        where :math:`N` is the batch size and :math:`S` is the sequence length.

    .. note::
        For a boolean padding mask, a ``True`` indicates that the corresponding
        position should be masked.
    """
    if seq_lens.dim() == 2 and seq_lens.size(1) != 1:
        raise ValueError("The size of the second dimension of `seq_len` must be 1.")

    if seq_lens.dim() >= 1:
        bsz = seq_lens.size(0)
    else:
        bsz = 1

    mask = torch.arange(max_seq_len, device=seq_lens.device)

    mask = mask.expand(bsz, -1) >= seq_lens.view(bsz, 1).expand(-1, max_seq_len)

    if seq_lens.dim() == 0:
        mask = mask.squeeze(0)

    return mask
