# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch import Tensor

from fairseq2.nn.ops import repeat_interleave
from fairseq2.typing import DataType, Device


def to_float_mask(mask: Tensor, dtype: DataType | None = None) -> Tensor:
    """Convert a boolean mask to a float mask.

    :param mask:
        The boolean mask. *Shape:* Any.
    :param dtype:
        The data type of the float mask. If ``None``, the default floating-point
        type will be used.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    return torch.zeros_like(mask, dtype=dtype).masked_fill_(mask, -torch.inf)


def compute_row_mask(
    shape: tuple[int, int],
    span_len: int,
    max_mask_prob: float,
    row_lens: Tensor | None = None,
    min_num_spans: int = 0,
    device: Device | None = None,
) -> Tensor | None:
    """Compute a random row mask of the specified shape.

    :param shape:
        The shape of the mask.
    :param span_len:
        The length of each mask span.
    :param max_mask_prob:
        The maximum probability of masking an element in a row. Note that, due
        to mask span overlap, the effective probability will be lower. The
        implementation also guarantees that there will be always at least one
        unmasked element in each row.
    :param row_lens:
        The length of each row. *Shape:* :math:`(R)`, where :math:`R` is the
        number of rows.
    :param min_num_spans:
        The minimum number of mask spans per row.
    :param device:
        The device on which to initialize the mask.

    :returns:
        The boolean row mask. *:Shape:* ``shape``.
    """
    num_rows, max_row_len = shape

    if row_lens is None:
        # We only mask rows that are longer than the mask span length.
        if span_len >= max_row_len:
            raise ValueError(
                f"The size of the second dimension of `shape` must be greater than `span_len` ({span_len}), but is {max_row_len} instead."
            )

        # (N)
        row_lens = torch.full(
            (num_rows,), max_row_len, device=device, dtype=torch.int64
        )
    else:
        # (N)
        row_lens = row_lens.view(num_rows)

        # We only mask rows that are longer than the mask span length.
        if (span_len >= row_lens).any():
            raise ValueError(
                f"All lengths in `row_lens` must be greater than `span_len` ({span_len}), but at least one length is smaller. row_lens: {row_lens}"
            )

    # (N, M x L)
    indices = _compute_mask_spans(row_lens, span_len, max_mask_prob, min_num_spans)
    if indices is None:
        return row_lens.new_empty((0, 0))

    return _generate_mask(indices, max_row_len).to(device)


def _compute_mask_spans(
    row_lens: Tensor, span_len: int, max_mask_prob: float, min_num_spans: int
) -> Tensor | None:
    """Compute random mask spans of the specified shape."""
    device, dtype = row_lens.device, row_lens.dtype

    num_rows = len(row_lens)
    if num_rows == 0:
        return None

    # Compute the number of mask spans per row. We should always have at least
    # one unmasked element; this is why we subtract 1 from `row_lens`.
    num_spans_per_row = max_mask_prob / span_len * (row_lens - 1)

    # Require the same number of mask spans for all rows.
    num_spans = int(num_spans_per_row.to(dtype).min())

    if min_num_spans > num_spans:
        raise ValueError(
            f"`min_num_spans` is {min_num_spans}, but with the given `span_len` and `max_mask_prob` only {num_spans} mask span(s) can be generated."
        )

    if num_spans == 0:
        return None

    # The range of possible start indices for mask spans in form [0, max + 1).
    # (N)
    span_start_range = row_lens - span_len + 1

    # (N) -> (N x M)
    span_start_range = repeat_interleave(span_start_range, dim=0, repeat=num_spans)

    # Unlike the fairseq implementation, we do sample with replacement, which is
    # more consistent with the overlap strategy.
    # (N x M)
    rand_scales = torch.rand(num_rows * num_spans, device=device)

    # By random scaling we effectively pick a random start index for each mask
    # span.
    span_offsets = span_start_range * rand_scales

    # The following ops convert the mask span offsets (i.e. start indices) to
    # mask spans (i.e. index ranges).
    # (N x M) -> (N, M)
    span_offsets = span_offsets.to(dtype).view(num_rows, -1)

    # (N, M) -> (N, M x L)
    span_offsets = repeat_interleave(span_offsets, dim=-1, repeat=span_len)

    # (L)
    indices = torch.arange(span_len, device=device, dtype=dtype)

    # (L) -> (N, M x L)
    indices = indices.repeat(num_spans).unsqueeze(0).expand(num_rows, -1)

    return span_offsets + indices


def _generate_mask(indices: Tensor, max_row_len: int) -> Tensor:
    """Generate a boolean mask by setting ``indices`` to ``True``."""
    # (N, S)
    float_mask = torch.zeros((indices.size(0), max_row_len), device=indices.device)

    # Set elements corresponding to masked indices to 1.
    float_mask.scatter_(1, indices, 1.0)

    # Since mask spans may overlap, rows might have varying number of masked
    # elements; therefore, we have to randomly unmask some of the elements to
    # ensure that all rows have the same amount of masking.
    min_num_masked = int(torch.count_nonzero(float_mask, dim=-1).min())

    # (N, min(M x L))
    # We randomly pick `min_num_masked` masked elements from each row, which
    # effectively unmasks the remaining elements.
    indices = torch.multinomial(float_mask, num_samples=min_num_masked)

    # (N, S)
    # Now we construct the actual boolean mask which has the same number of
    # masked elements in each row.
    bool_mask = torch.full_like(float_mask, False, dtype=torch.bool)

    return bool_mask.scatter_(1, indices, True)


def get_num_masks(mask: Tensor) -> Tensor:
    bsz, seq_len = mask.shape

    # Pad mask with False at the end of each sequence to handle trailing True values
    padded_mask = torch.cat(
        [mask, torch.zeros(bsz, 1, dtype=torch.bool, device=mask.device)], dim=1
    )

    # We are interested in the transitions from True to False, which are the ends of the masks
    ends_of_masks = (mask & ~padded_mask[:, 1:]).float()

    # Count the number of True to False transitions, which equals the number of True groups
    num_masks = ends_of_masks.sum()

    return num_masks
