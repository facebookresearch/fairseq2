# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, cast

import torch
from torch import Tensor


def compute_mask(
    shape: Tuple[int, int],
    mask_len: int,
    mask_prob: float,
    row_lens: Optional[Tensor] = None,
    min_num_masks: int = 0,
    device: Optional[torch.device] = None,
) -> Optional[Tensor]:
    """Compute a random mask for the specified shape.

    :param shape:
        The two dimensional shape for which to compute a mask.
    :param mask_len:
        The length of each mask span.
    :param mask_prob:
        The probability of masking an element among all elements in a particular
        row. Note that, due to span overlap, the actual number might be smaller.
    :param row_lens:
        The length of each row if ``shape`` is ragged.
    :param min_num_masks:
        The minimum number of mask spans per row.
    :param device:
        The device on which to initialize the mask.

    :returns:
        A boolean mask. *:Shape:* ``shape``.

    .. note::
        For a boolean padding mask, a ``True`` indicates that the corresponding
        position should be masked.
    """
    num_rows, max_row_len = shape

    if row_lens is None:
        # We only mask rows that are longer than the mask span length.
        if mask_len >= max_row_len:
            return None

        row_lens = torch.full(
            (num_rows,), max_row_len, device=device, dtype=torch.int64
        )

        effective_row_mask = None
    else:
        row_lens = row_lens.view(num_rows)

        # We only mask rows that are longer than the mask span length.
        effective_row_mask = (row_lens - mask_len) > 0

        # Exclude rows that we won't to mask.
        row_lens = row_lens[effective_row_mask]

    indices = _compute_mask_indices(row_lens, mask_len, mask_prob, min_num_masks)

    # If `None`, it means we won't be masking any elements.
    if indices is None:
        return None

    mask = torch.full(shape, False, device=row_lens.device, dtype=torch.bool)

    # Set elements corresponding to masked indices to `True`.
    if effective_row_mask is None:
        mask.scatter_(1, indices, True)
    else:
        num_sub_rows = indices.size(0)

        # Apply indices to a sub-mask first since not all rows are masked.
        sub_mask = torch.full(
            (num_sub_rows, max_row_len), False, device=row_lens.device, dtype=torch.bool
        )

        mask[effective_row_mask] = sub_mask.scatter_(1, indices, True)

    return mask.to(device)


def _compute_mask_indices(
    row_lens: Tensor, mask_len: int, mask_prob: float, min_num_masks: int
) -> Optional[Tensor]:
    """Compute random mask spans for the specified ragged shape."""
    device, dtype = row_lens.device, row_lens.dtype

    num_rows = row_lens.size(0)
    if num_rows == 0:
        return None

    # Used for probabilistic rounding between floor and ceil.
    rounding = torch.rand(num_rows, device=device)

    # Compute the number of masks per row. We should always have at least one
    # unmasked element; that is why we substract 1 from `row_lens`.
    num_masks_per_row = ((mask_prob / mask_len) * (row_lens - 1)) + rounding

    # Require the same number of masks for all rows.
    num_masks = cast(int, num_masks_per_row.type(dtype).min().item())

    if min_num_masks > num_masks:
        num_masks = min_num_masks

    if num_masks == 0:
        return None

    # The range of possible start indices for masks in form of [0, max + 1).
    mask_start_range = row_lens - mask_len + 1

    # (R) -> (R x N)
    mask_start_range = mask_start_range.repeat_interleave(num_masks)

    # Unlike the fairseq implementation, we do sample with replacement, which is
    # more consistent with the overlap strategy.
    # (R x N)
    random_scales = torch.rand(num_rows * num_masks, device=device)

    # By random scaling we effectively pick a random start index for each mask.
    mask_offsets = mask_start_range * random_scales

    # The following operations convert the mask offsets (i.e. start indices) to
    # mask spans (i.e. index ranges).
    # (R x N) -> (R, N)
    mask_offsets = mask_offsets.type(dtype).view(num_rows, -1)

    # (R, N) -> (R, N x L)
    mask_offsets = mask_offsets.repeat_interleave(mask_len, dim=-1)

    # (L)
    indices = torch.arange(mask_len, device=device, dtype=dtype)

    # (L) -> (R, N x L)
    indices = indices.repeat(num_masks).unsqueeze(0).expand(num_rows, -1)

    return mask_offsets + indices


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
