# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, final

import torch
from torch import Tensor
from torch.nn.functional import pad

from fairseq2.typing import finaloverride


class StepProcessor(ABC):
    """Processes next-step probabilities during sequence generation."""

    @abstractmethod
    def __call__(self, seqs: Tensor, probs: Tensor, lprob: bool = False) -> None:
        """
        :param seqs:
            The sequences that are in process of being generated. *Shape:*
            :math:`(N,S)`, where :math`N` is the batch size and :math:`S` is the
            sequence length generated so far.
        :param probs:
            The next-step probabilities of ``seqs``. *Shape:* :math:`(N,V)`,
            where :math:`N` is the batch size and :math:`V` is the size of the
            target vocabulary.
        :param lprob:
            If ``True``, ``probs`` contains log probabilities.
        """


@final
class BannedSequenceProcessor(StepProcessor):
    """Prevents a provided list of banned sequences from being generated."""

    _banned_seqs: Optional[Tensor]
    _banned_mask: Optional[Tensor]

    def __init__(self, banned_seqs: Sequence[Tensor]) -> None:
        """
        :param banned_seqs:
            The list of banned sequences.
        """
        batch_size = len(banned_seqs)

        if batch_size == 0:
            self._banned_seqs = None
            self._banned_mask = None

            return

        max_seq_len = 0
        min_seq_len = sys.maxsize

        seq_lens: List[int] = []

        for idx, seq in enumerate(banned_seqs):
            seq_len = len(seq)
            if seq_len == 0:
                raise ValueError(f"`banned_seqs[{idx}]` must not be empty.")

            seq_lens.append(seq_len)

            max_seq_len = max(seq_len, max_seq_len)
            min_seq_len = min(seq_len, min_seq_len)

        device = banned_seqs[0].device

        # (N, S)
        self._banned_seqs = torch.zeros(
            (batch_size, max_seq_len), device=device, dtype=torch.int64
        )

        if max_seq_len != min_seq_len:
            # (N, S)
            self._banned_mask = torch.full(
                (batch_size, max_seq_len), True, device=device
            )
        else:
            self._banned_mask = None

        for row, seq in enumerate(banned_seqs):
            if self._banned_mask is None:
                self._banned_seqs[row] = seq
            else:
                self._banned_seqs[row, -seq_lens[row] :] = seq
                self._banned_mask[row, -seq_lens[row] :] = False

    @finaloverride
    def __call__(self, seqs: Tensor, probs: Tensor, lprob: bool = False) -> None:
        if self._banned_seqs is None:
            return

        ban_value = -torch.inf if lprob else 0

        banned_prefix_len = self._banned_seqs.size(1) - 1
        if banned_prefix_len == 0:
            probs[:, self._banned_seqs[:, 0]] = ban_value

            return

        if (len_delta := banned_prefix_len - seqs.size(1)) > 0:
            # (N, S) -> (N, S_pre)
            seqs = pad(seqs, (len_delta, 0), value=-1)
        elif len_delta < 0:
            # (N, S) -> (N, S_pre)
            seqs = seqs[:, -banned_prefix_len:]

        # (N, S_pre) -> (N, 1, S_pre)
        seqs = seqs.unsqueeze(1)

        # (N, 1, S_pre) - (B, S_pre) -> (N, B, S_pre)
        seqs = seqs - self._banned_seqs[:, :-1]

        if self._banned_mask is not None:
            seqs.masked_fill_(self._banned_mask[:, :-1], 0)

        # (N, B, S_pre) -> (N, B)
        banned_prefix_matches = seqs.sum(dim=-1)

        # (N, B) -> (N), (B)
        batch_indices, banned_indices = torch.where(banned_prefix_matches == 0)

        if len(batch_indices) > 0:
            probs[batch_indices, self._banned_seqs[:, -1][banned_indices]] = ban_value
