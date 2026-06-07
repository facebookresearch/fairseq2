# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers for translating ``fairseq2.nn.BatchLayout`` into shapes consumed by
the ``mamba_ssm`` Triton kernels.

Mamba2's selective scan needs to know where one logical sequence ends and the
next begins inside a *packed* batch so it can reset the SSM state. fairseq2
encodes packing/padding via :class:`BatchLayout`; mamba_ssm encodes it via the
``seq_idx`` tensor: shape ``[N, S]`` of int32 where each entry is the
0-indexed sub-sequence id of that token (and padding positions can carry any
value the kernel is told to ignore, in practice the last valid id).

This module is the single seam between the two representations. Keeping it
isolated means the Mamba2 mixer code never has to think about packing.
"""

from __future__ import annotations

import torch
from torch import Tensor

from fairseq2.nn import BatchLayout


def batch_layout_to_seq_idx(layout: BatchLayout) -> Tensor | None:
    """Translate a :class:`BatchLayout` into a mamba_ssm ``seq_idx`` tensor.

    :param layout:
        The batch layout describing the shape of the input.

    :returns:
        - ``None`` when every logical sequence occupies its own row (whether
          or not those rows are padded to a common width). The SSM recurrence
          is then identical to running each row independently; mamba_ssm's
          ``seq_idx=None`` path is correct and fastest. Padding tokens still
          enter the recurrence but their outputs are masked by the loss /
          downstream attention, matching the established convention used for
          regular self-attention with padding.
        - An ``int32`` tensor of shape ``[1, sum(seq_lens)]`` for a *packed*
          batch (multiple logical sequences concatenated into a single row).
          Entry ``[0, t]`` is the 0-indexed sub-sequence id of token ``t``,
          so the kernel resets the SSM state at every sub-sequence boundary.

    .. note::
        The returned tensor lives on the same device as
        ``layout.seq_lens_pt`` and is contiguous, which is what
        ``mamba_chunk_scan_combined`` expects.
    """
    # Only the packed-multi-sequence case truly needs seq_idx. For ordinary
    # rectangular [N, S] batches and padded-but-one-sequence-per-row batches,
    # returning None lets the kernel take its faster default path. Padding
    # tokens still propagate state, which matches what fairseq2's MHA does
    # with padded queries -- the outputs are masked by the loss, not by the
    # mixer.
    if not layout.packed:
        return None

    seq_lens = layout.seq_lens  # Sequence[int]
    device = layout.seq_lens_pt.device

    # Packed batch: input is a single concatenated row of width sum(seq_lens).
    # Emit one row of [0,0,...,1,1,...,K-1,...] using repeat_interleave for
    # vectorised construction.
    total = sum(seq_lens)
    ids = torch.arange(len(seq_lens), dtype=torch.int32, device=device)
    seq_idx = ids.repeat_interleave(layout.seq_lens_pt)
    assert seq_idx.shape == (
        total,
    ), f"packed seq_idx width mismatch: got {seq_idx.shape}, expected ({total},)"
    return seq_idx.unsqueeze(0).contiguous()
