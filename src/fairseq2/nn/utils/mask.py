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
    span_len: int,
    max_mask_prob: float,
    row_lens: Optional[Tensor] = None,
    min_num_spans: int = 0,
    device: Optional[torch.device] = None,
) -> Optional[Tensor]:
    """Compute a random mask for the specified shape.

    :param shape:
        The two dimensional shape for which to compute a mask.
    :param span_len:
        The length of each mask span in a row. Note that rows whose length is
        less than or equal to ``span_len`` will never be masked.
    :param max_mask_prob:
        The maximum probability of masking an element among all elements in a
        particular row. Note that, due to mask span overlap, the effective
        probability might be smaller. The implementation also guarantees that
        there is always at least one unmasked element in each row.
    :param row_lens:
        The length of each row if ``shape`` is ragged.
    :param min_num_spans:
        The minimum number of mask spans per row.
    :param device:
        The device on which to initialize the mask.

    :returns:
        A boolean mask. *:Shape:* ``shape``.

    .. note::
        For a boolean mask, a ``True`` indicates that the corresponding element
        should be masked.
    """
    num_rows, max_row_len = shape

    if row_lens is None:
        # We only mask rows that are longer than the mask span length.
        if span_len >= max_row_len:
            return None

        row_lens = torch.full(
            (num_rows,), max_row_len, device=device, dtype=torch.int64
        )

        valid_rows_mask = None
    else:
        row_lens = row_lens.view(num_rows)

        # We only mask rows that are longer than the mask span length.
        valid_rows_mask = (row_lens - span_len) > 0

        # Filter out rows that we won't to mask.
        row_lens = row_lens[valid_rows_mask]

    indices = _compute_mask_spans(row_lens, span_len, max_mask_prob, min_num_spans)

    # If `None`, it means we won't mask any elements.
    if indices is None:
        return None

    mask = _generate_mask(indices, max_row_len)

    # We have to include rows that were shorter than the span length, and
    # therefore were filtered out, in the final mask as unmasked.
    if valid_rows_mask is not None:
        tmp = torch.full(shape, False, device=indices.device, dtype=torch.bool)

        tmp[valid_rows_mask] = mask

        mask = tmp

    return mask.to(device)


def _compute_mask_spans(
    row_lens: Tensor, span_len: int, max_mask_prob: float, min_num_spans: int
) -> Optional[Tensor]:
    """Compute random mask spans for the specified (ragged) shape."""
    device, dtype = row_lens.device, row_lens.dtype

    num_rows = row_lens.size(0)
    if num_rows == 0:
        return None

    # Used for probabilistic rounding between floor and ceil.
    rounding = torch.rand(num_rows, device=device)

    # Compute the number of mask spans per row. We should always have at least
    # one unmasked element; that is why we substract 1 from `row_lens`.
    num_spans_per_row = ((max_mask_prob / span_len) * (row_lens - 1)) + rounding

    # Require the same number of mask spans for all rows.
    num_spans = cast(int, num_spans_per_row.type(dtype).min().item())

    if min_num_spans > num_spans:
        num_spans = min_num_spans

    if num_spans == 0:
        return None

    # The range of possible start indices for mask spans in form [0, max + 1).
    span_start_range = row_lens - span_len + 1

    # (R) -> (R x N)
    span_start_range = span_start_range.repeat_interleave(num_spans)

    # Unlike the fairseq implementation, we do sample with replacement, which is
    # more consistent with the overlap strategy.
    # (R x N)
    random_scales = torch.rand(num_rows * num_spans, device=device)

    # By random scaling we effectively pick a start index for each mask span.
    span_offsets = span_start_range * random_scales

    # The following ops convert the mask span offsets (i.e. start indices) to
    # mask spans (i.e. index ranges).
    # (R x N) -> (R, N)
    span_offsets = span_offsets.type(dtype).view(num_rows, -1)

    # (R, N) -> (R, N x L)
    span_offsets = span_offsets.repeat_interleave(span_len, dim=-1)

    # (L)
    indices = torch.arange(span_len, device=device, dtype=dtype)

    # (L) -> (R, N x L)
    indices = indices.repeat(num_spans).unsqueeze(0).expand(num_rows, -1)

    return span_offsets + indices


def _generate_mask(indices: Tensor, max_row_len: int) -> Tensor:
    """Generates a boolean mask by masking ``indices``."""
    float_mask = torch.zeros((indices.size(0), max_row_len), device=indices.device)

    # Set elements corresponding to masked indices to 1.
    float_mask.scatter_(1, indices, 1.0)

    # Since mask spans may overlap, rows might have varying number of masked
    # elements; therefore, we have to randomly unmask some of the elements to
    # ensure that all rows have the same amount of masking.
    min_num_masked = cast(int, torch.count_nonzero(float_mask, dim=-1).min().item())

    # We randomly pick `min_num_masked` masked elements from each row, which
    # effectively unmasks the remaining elements.
    indices = torch.multinomial(float_mask, num_samples=min_num_masked)

    # Now we construct the actual boolean mask which has the same number of
    # masked elements in each row.
    bool_mask = torch.full_like(float_mask, False, dtype=torch.bool)

    return bool_mask.scatter_(1, indices, True)


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
