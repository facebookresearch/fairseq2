# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor


def to_padding_mask(seq_lens: Tensor, mask_seq_len: int) -> Tensor:
    """Convert a sequence length array to a boolean padding mask.

    :param seq_lens:
        An array where each element represents the length of a sequence.
        *Shape:* :math:`(N)`, :math:`(N,1)`, or :math:`()` when unbatched, where
        :math:`N` is the batch size.
    :param mask_seq_len:
        The sequence length of the returned padding mask.

    :returns:
        The padding mask. *Shape:* :math:`(N,S)`, or :math:`(S)` when unbatched,
        where :math:`N` is the batch size and :math:`S` is ``mask_seq_len``.

    .. note::
        For a boolean padding mask, a ``True`` indicates that the corresponding
        position should be masked.
    """
    if seq_lens.dim() == 2 and seq_lens.size(1) != 1:
        raise ValueError(
            f"The size of the second dimension of `seq_lens` must be 1 when it is two dimensional, but is {seq_lens.size(1)} instead."
        )

    if seq_lens.dim() >= 1:
        batch_size = seq_lens.size(0)
    else:
        batch_size = 1

    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)

    mask = indices >= seq_lens.view(batch_size, 1).expand(-1, mask_seq_len)

    if seq_lens.dim() == 0:
        mask = mask.squeeze(0)

    return mask


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
