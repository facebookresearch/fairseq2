# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from fairseq2.device import Device
from fairseq2.ops import repeat_interleave, unsqueeze


def apply_mask(
    seqs: Tensor, mask: Tensor, *, fill_value: int | float | Tensor = 0
) -> Tensor:
    """
    Applies the specified boolean mask to ``seqs``.

    :param seqs: The sequences to mask. *Shape:* :math:`(N,S,*)`, where :math:`N`
        is the batch size, :math:`S` is the sequence length, and :math:`*` is
        any number of sequence-specific dimensions including none.
    :param mask: The boolean mask.

    :returns: The input sequences with mask applied. *Shape:* Same as ``seqs``.
    """
    mask = unsqueeze(mask, dim=-1, count=seqs.ndim - mask.ndim)

    return seqs.where(mask, fill_value)


class RowMaskFactory(Protocol):
    def __call__(
        self,
        shape: tuple[int, int],
        span_len: int,
        max_mask_prob: float,
        row_lens: Tensor | None = None,
        min_num_spans: int = 0,
        device: Device | None = None,
    ) -> Tensor | None:
        """
        Computes a random row mask of the specified shape.

        :param shape:
            The shape of the mask.
        :param span_len:
            The length of each mask span.
        :param max_mask_prob:
            The maximum probability of masking an element in a row.
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


def compute_row_mask(
    shape: tuple[int, int],
    span_len: int,
    max_mask_prob: float,
    row_lens: Tensor | None = None,
    min_num_spans: int = 0,
    device: Device | None = None,
) -> Tensor | None:
    """
    Implements the :class:`RowMaskFactory` protocol.

    Note that, due to mask span overlap, the effective mask probability will be
    lower than ``max_mask_prob``. The implementation also guarantees that there
    will be always at least one unmasked element in each row.
    """
    num_rows, max_row_len = shape

    if row_lens is None:
        # We only mask rows that are longer than the mask span length.
        if span_len >= max_row_len:
            raise ValueError(
                f"Size of the second dimension of `shape` must be greater than `span_len` ({span_len}), but is {max_row_len} instead."
            )

        # (N)
        row_lens = torch.full(
            (num_rows,), max_row_len, device=device, dtype=torch.int64
        )
    else:
        # (N)
        row_lens = row_lens.to(torch.int64).view(num_rows)

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
    #
    # We first make a tensor of random values and 0.001 to it to ensure the
    # minimum value is larger than 0. Then we multiply it with the float_mask so
    # that all the 0 values in `float_mask` are still 0 but the non-zero values
    # have a random value assigned to them. Then we select the top-k values,
    #  which would be basically a subset of non-zero values `float_mask`.
    random_values = torch.rand_like(float_mask) + 0.001

    random_values = random_values * float_mask

    _, indices = torch.topk(random_values, k=min_num_masked, dim=1, sorted=False)

    # (N, S)
    # Now we construct the actual boolean mask which has the same number of
    # masked elements in each row.
    bool_mask = torch.full_like(float_mask, False, dtype=torch.bool)

    return bool_mask.scatter_(1, indices, True)
