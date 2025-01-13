# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import cast, final

import torch
from torch import Tensor

from fairseq2.data import Collater, SequenceData
from fairseq2.typing import Device


@final
class PaddingMask:
    """Represents a sequence padding mask."""

    _seq_lens: Tensor
    _batch_seq_len: int
    _materialized: Tensor | None
    _materialized_float: Tensor | None

    def __init__(self, seq_lens: Tensor, batch_seq_len: int) -> None:
        """
        :param seq_lens:
            An array where each element represents the length of a sequence.
            *Shape:* :math:`(N)`, where :math:`N` is the batch size.
        :param batch_seq_len:
            The sequence length of the mask.
        """
        self._seq_lens = seq_lens
        self._batch_seq_len = batch_seq_len

        self._materialized = None
        self._materialized_float = None

    def materialize(self) -> Tensor:
        """Materialize the padding mask as a boolean tensor."""
        if self._materialized is None:
            self._materialized = to_padding_mask(self._seq_lens, self._batch_seq_len)

        return self._materialized

    def materialize_as(self, seqs: Tensor) -> Tensor:
        """Materialize the padding mask as a float tensor."""
        if self._materialized_float is None:
            bool_mask = self.materialize()

            # (N, S)
            mask = torch.zeros_like(bool_mask, dtype=seqs.dtype)

            mask = torch.where(bool_mask, mask, -torch.inf)

            self._materialized_float = mask

        return self._materialized_float

    def trim(self, size: int) -> PaddingMask:
        """Return a new trimmed padding mask.

        :param size:
            The amount by which to trim the sequences.
        """
        return PaddingMask(self._seq_lens - size, self._batch_seq_len - size)

    def to(self, device: Device) -> PaddingMask:
        """Perform device conversion.

        :param device:
            The target device.
        """
        if self._seq_lens.device == device:
            return self

        return PaddingMask(self._seq_lens.to(device), self._batch_seq_len)

    @property
    def seq_lens(self) -> Tensor:
        """An array where each element represents the length of a sequence.
        *Shape:* :math:`(N)`, where :math:`N` is the batch size."""
        return self._seq_lens


def to_padding_mask(seq_lens: Tensor, batch_seq_len: int) -> Tensor:
    """Convert a sequence length array to a boolean padding mask tensor.

    :param seq_lens:
        An array where each element represents the length of a sequence. *Shape:*
        :math:`(N)`, where :math:`N` is the batch size.
    :param batch_seq_len:
        The sequence length of the mask.

    :returns:
        The mask. *Shape:* :math:`(N,S)`, where :math:`N` is the batch size and
        :math:`S` is the sequence length.
    """
    batch_size = seq_lens.size(0)

    # (N, S)
    indices = torch.arange(batch_seq_len, device=seq_lens.device).expand(batch_size, -1)

    # (N) -> (N, S)
    lengths = seq_lens.unsqueeze(1).expand(-1, batch_seq_len)

    return indices < lengths


def get_seq_lens(seqs: Tensor, padding_mask: PaddingMask | None) -> Tensor:
    """Retrieve the sequence lengths of ``seqs``.

    :param seqs:
        The sequences. *Shape:* :math:`(N,S,*)`, where :math:`N` is the batch
        size, :math:`S` is the sequence length, and :math:`*` is any number of
        sequence-specific dimensions including none.
    :param padding_mask:
        The padding mask. *Shape:* :math:`(N,S)`, where :math:`N` is the batch
        size and :math:`S` is the sequence length.

    :returns:
        An array where each element represents the length of the corresponding
        sequence in ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is the batch
        size.
    """
    if padding_mask is not None:
        return padding_mask.seq_lens

    return torch.full((seqs.size(0),), seqs.size(1), device=seqs.device)


def apply_padding_mask(
    seqs: Tensor, padding_mask: PaddingMask | None, pad_value: int | float | Tensor = 0
) -> Tensor:
    """Apply the specified padding mask to ``seqs``.

    :param seqs:
        The sequences to mask. *Shape:* :math:`(N,S,*)`, where :math:`N` is the
        batch size, :math:`S` is the sequence length, and :math:`*` is any
        number of sequence-specific dimensions including none.
    :param padding_mask:
        The padding mask to apply. *Shape:* :math:`(N,S)`, where :math:`N` is
        the batch size and :math:`S` is the sequence length.
    :param pad_value:
        The value for padded positions.

    :returns:
        The input sequences with mask applied. *Shape:* Same as ``seqs``.
    """
    if padding_mask is None:
        return seqs

    m = padding_mask.materialize()

    for _ in range(seqs.ndim - m.ndim):
        m = m.unsqueeze(-1)

    return seqs.where(m, pad_value)


def get_seqs_and_padding_mask(
    data: SequenceData, device: Device | None = None
) -> tuple[Tensor, PaddingMask | None]:
    """Return the sequences along with their padding mask from ``data``.

    :returns:
        - The sequences (i.e. `data["seqs"]`)
        - The padding mask of the returned sequences.
    """
    seqs = data["seqs"]

    if device is not None:
        seqs = seqs.to(device)

    if not data["is_ragged"]:
        return seqs, None

    seq_lens = data["seq_lens"]

    if device is not None:
        seq_lens = seq_lens.to(device)

    return seqs, PaddingMask(seq_lens, batch_seq_len=seqs.size(1))


def pad_seqs(
    seqs: Sequence[Tensor], pad_value: int = 0, pad_to_multiple: int = 1
) -> tuple[Tensor, PaddingMask | None]:
    """Stack ``seqs`` along a new batch dimension and pad them to equal length.

    :param seqs:
        The list of variable length sequences. All elements in ``seqs`` are
        expected to have the same shape except the first dimension.
    :param pad_value:
        The value for padded positions.
    :param pad_to_multiple:
        The sequence dimension is rounded up to the nearest multiple of the
        specified value.

    :returns:
        - The padded sequence stack. *Shape:* :math:`(N,S,*)`, where :math:`N`
          is the batch size, :math:`S` is the sequence length, and :math:`*` is
          any number of sequence-specific dimensions including none.
        - The padding mask of the sequence stack. *Shape:* :math:`(N,S)`, where
          :math:`N` is the batch size and :math:`S` is the sequence length.
    """
    collater = Collater(pad_value=pad_value, pad_to_multiple=pad_to_multiple)

    seq_data = cast(SequenceData, collater(seqs))

    return get_seqs_and_padding_mask(seq_data)
