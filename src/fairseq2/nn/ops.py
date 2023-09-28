# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence, Tuple, cast

from torch import Tensor

from fairseq2.data.data_pipeline import Collater, SequenceData


def pad_sequence(
    seqs: Sequence[Tensor], pad_idx: int = 0, pad_to_multiple: int = 1
) -> Tuple[Tensor, Optional[Tensor]]:
    """Stack ``seqs`` along the first dimension and pad them to equal length.

    :param seqs:
        The list of variable length sequences. All elements in ``seqs`` are
        expected to have the same shape except the first dimension.
    :param pad_idx:
        The value for padded positions.
    :param pad_to_multiple:
        The sequence dimension is rounded up to the nearest multiple of the
        specified value.

    :returns:
        - The padded sequence stack. *Shape:* :math:`(N,S,*)`, where :math:`N`
          is the batch size, :math:`S` is the sequence length, and :math:`*` is
          any number of sequence-specific dimensions including none.
        - An array where each element represents the length of the sequence at
          the same index in the first returned value. *Shape:* :math:`(N)`,
          where :math:`N` is the batch size. ``None`` if all input sequences
          have the same length and it is a multiple of ``pad_to_multiple``.
    """
    collater = Collater(pad_idx=pad_idx, pad_to_multiple=pad_to_multiple)

    output = cast(SequenceData, collater(seqs))

    padded_seqs, seq_lens = output["seqs"], output["seq_lens"]

    if not output["is_ragged"]:
        # If `pad_to_multiple` is greater than 1, we might still need to return
        # the sequence lengths even if all sequences have the same length.
        if padded_seqs.size(1) == seq_lens[0]:
            return padded_seqs, None

    return padded_seqs, seq_lens


def repeat_interleave(x: Tensor, dim: int, repeat: int) -> Tensor:
    """Repeat elements of a tensor.

    :param x:
        The input tensor.
    :param dim:
        The dimension along which to repeat values.
    :param repeat:
        The number of repetitions.

    :returns:
        The repeated tensor which has the same shape as input, except along the
        given axis.

    .. note::
        This is a lightweight version of :func:`torch.repeat_interleave` that
        is faster for repetitions along a single dimension.
    """
    if repeat == 1:
        return x

    shape = [-1] * (x.ndim + 1)

    if dim < 0:
        dim += x.ndim

    shape[dim + 1] = repeat

    return x.unsqueeze(dim + 1).expand(shape).flatten(dim, dim + 1)
