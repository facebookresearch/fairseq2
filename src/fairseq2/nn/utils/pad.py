# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor


def to_padding_mask(seq_lens: Tensor, max_seq_len: int) -> Tensor:
    """Convert a sequence length array into a boolean padding mask.

    :param seq_lens:
        The sequence length array. Each entry defines the length of the sequence
        at the same index in the corresponding mini-batch. *Shape:* :math:`(N)`,
        :math:`(N,1)`, or :math:`()` when unbatched, where :math:`N` is the
        batch size.
    :param max_seq_len:
        The sequence length of the returned padding mask.

    :returns:
        The padding mask. *Shape:* :math:`(N,S)`, or :math:`(S)` when unbatched,
        where :math:`N` is the batch size and :math:`S` is the sequence length.

    .. note::
        For a boolean padding mask, a ``True`` indicates that the corresponding
        position should be masked.
    """
    if seq_lens.dim() == 2 and seq_lens.size(1) != 1:
        raise ValueError(
            f"The size of the second dimension of `seq_lens` must be 1 when it is two dimensional, but is {seq_lens.size(1)} instead."
        )

    if seq_lens.dim() >= 1:
        bsz = seq_lens.size(0)
    else:
        bsz = 1

    ind = torch.arange(max_seq_len, device=seq_lens.device).expand(bsz, -1)

    mask = ind >= seq_lens.view(bsz, 1).expand(-1, max_seq_len)

    if seq_lens.dim() == 0:
        mask = mask.squeeze(0)

    return mask
