# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, cast

import torch
from torch import Tensor

from fairseq2.typing import DataType, Device


def to_padding_mask(seqs: Tensor, seq_lens: Optional[Tensor]) -> Optional[Tensor]:
    """Convert a sequence length array to a float padding mask.

    :param seqs:
        The sequences to mask. *Shape:* :math:`(N,S,*)`, where :math:`N` is the
        batch size, :math:`S` is the sequence length, and :math:`*` is any
        number of sequence-specific dimensions including none.
    :param seq_lens:
        An array where each element represents the length of the sequence at the
        same index in ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is the
        batch size.

    :returns:
        The float padding mask. *Shape:* :math:`(N,S)`, where :math:`N` is the
        batch size and :math:`S` is the sequence length.
    """
    if seq_lens is None:
        return None

    batch_size, mask_seq_len = seqs.shape[:2]

    # No need to construct a mask if all sequences have the same length.
    if (seq_lens == mask_seq_len).all():
        return None

    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)

    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)

    mask = seqs.new_zeros((batch_size, mask_seq_len))

    mask.masked_fill_(bool_mask, -torch.inf)

    return mask


def to_float_mask(mask: Tensor, dtype: DataType = torch.float32) -> Tensor:
    """Convert a boolean mask to a float mask.

    :param mask:
        The mask tensor. *Shape:* Any.
    :param dtype:
        The floating-point type of the converted mask.
    """
    return torch.zeros_like(mask, dtype=dtype).masked_fill_(mask, -torch.inf)


def apply_padding_mask(seqs: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
    """Apply the specified padding mask to ``seqs``.

    :param seqs:
        The sequences to mask. *Shape:* :math:`(N,S,*)`, where :math:`N` is the
        the batch size, :math:`S` is the sequence length, and :math:`*` is any
        number of sequence-specific dimensions including none.
    :param padding_mask:
        The float padding mask to apply. *Shape:* :math:`(N_{msk},S)`, where
        :math:`N_{msk}` is the mask batch size and :math:`S` is the sequence
        length. :math:`N` can be a multiple of :math:`N_{msk}` in which case the
        mask will be tiled before being applied.

    :returns:
        The input sequences with mask applied. *Shape:* Same as ``seqs``.
    """
    if padding_mask is None:
        return seqs

    bool_mask = padding_mask.isinf()

    seq_batch_size, mask_batch_size = seqs.size(0), padding_mask.size(0)

    if seq_batch_size != mask_batch_size:
        bool_mask = bool_mask.repeat(seq_batch_size // mask_batch_size, 1)

    return seqs.masked_fill(bool_mask.unsqueeze(2), 0.0)


def compute_mask(
    shape: Tuple[int, int],
    span_len: int,
    max_mask_prob: float,
    row_lens: Optional[Tensor] = None,
    min_num_spans: int = 0,
    device: Optional[Device] = None,
) -> Optional[Tensor]:
    """Compute a random mask for the specified shape.

    :param shape:
        The two dimensional shape for which to compute a mask.
    :param span_len:
        The length of each mask span.
    :param max_mask_prob:
        The maximum probability of masking an element among all elements in a
        row. Note that, due to mask span overlap, the effective probability
        might be smaller. The implementation also guarantees that there is
        always at least one unmasked element in each row.
    :param row_lens:
        The length of each row if ``shape`` is ragged.
    :param min_num_spans:
        The minimum number of mask spans per row.
    :param device:
        The device on which to initialize the mask.

    :returns:
        A boolean mask. *:Shape:* ``shape``.
    """
    num_rows, max_row_len = shape

    if row_lens is None:
        # We only mask rows that are longer than the mask span length.
        if span_len >= max_row_len:
            raise ValueError(
                f"The size of the second dimension of `shape` must be greater than {span_len}, but is {max_row_len} instead."
            )

        row_lens = torch.full(
            (num_rows,), max_row_len, device=device, dtype=torch.int64
        )
    else:
        row_lens = row_lens.view(num_rows)

        # We only mask rows that are longer than the mask span length.
        if (span_len >= row_lens).any():
            raise ValueError(
                f"All lengths in `row_lens` must be greater than {span_len}, but at least one length is smaller. row_lens: {row_lens}"
            )

    indices = _compute_mask_spans(row_lens, span_len, max_mask_prob, min_num_spans)

    if indices is None:
        return row_lens.new_empty((0, 0))

    return _generate_mask(indices, max_row_len).to(device)


def _compute_mask_spans(
    row_lens: Tensor, span_len: int, max_mask_prob: float, min_num_spans: int
) -> Optional[Tensor]:
    """Compute random mask spans for the specified (ragged) shape."""
    device, dtype = row_lens.device, row_lens.dtype

    num_rows = row_lens.size(0)
    if num_rows == 0:
        return None

    # Compute the number of mask spans per row. We should always have at least
    # one unmasked element; this is why we substract 1 from `row_lens`.
    num_spans_per_row = (max_mask_prob / span_len) * (row_lens - 1)

    # Require the same number of mask spans for all rows.
    num_spans = cast(int, num_spans_per_row.type(dtype).min().item())

    if min_num_spans > num_spans:
        raise ValueError(
            f"`min_num_spans` is {min_num_spans}, but with the given `span_len` and `max_mask_prob` only {num_spans} mask span(s) can be generated."
        )

    if num_spans == 0:
        return None

    # The range of possible start indices for mask spans in form [0, max + 1).
    span_start_range = row_lens - span_len + 1

    # (R) -> (R x N)
    span_start_range = span_start_range.repeat_interleave(num_spans)

    # Unlike the fairseq implementation, we do sample with replacement, which is
    # more consistent with the overlap strategy.
    # (R x N)
    rand_scales = torch.rand(num_rows * num_spans, device=device)

    # By random scaling we effectively pick a random start index for each mask
    # span.
    span_offsets = span_start_range * rand_scales

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
    """Generate a boolean mask by setting ``indices`` to ``True``."""
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
