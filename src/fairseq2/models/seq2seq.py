# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import log_softmax

from fairseq2.nn.functional import nll_loss


class Seq2SeqModel(Module, ABC):
    """Represents a sequence-to-sequence model."""

    @abstractmethod
    def forward(
        self,
        source_seqs: Tensor,
        source_seq_lens: Optional[Tensor],
        target_seqs: Tensor,
        target_seq_lens: Optional[Tensor],
    ) -> "Seq2SeqModelOutput":
        """
        :param source_seqs:
            The source sequences. *Shape:* :math:`(N,S_{src},*)`, where
            :math:`N` is the batch size, :math:`S_{src}` is the source sequence
            length, and :math:`*` is any number of sequence-specific dimensions
            including none.
        :param source_seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``source_seqs``. *Shape:* :math:`(N)`, where
            :math:`N` is the batch size.
        :param target_seqs:
            The target sequences. *Shape:* :math:`(N,S_{tgt},*)`, where
            :math:`N` is the batch size, :math:`S_{tgt}` is the target sequence
            length, and :math:`*` is any number of sequence-specific dimensions
            including none.
        :param target_seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``target_seqs``. *Shape:* :math:`(N)`, where
            :math:`N` is the batch size.
        """


@dataclass
class Seq2SeqModelOutput:
    """Holds the output of a sequence-to-sequence model."""

    logits: Tensor
    """The logits for next-step prediction. *Shape:* :math:`(N,S_{tgt},T)`,
    where :math:`N` is the batch size, :math:`S_{tgt}` is the target sequence
    length, and :math:`T` is the size of the target domain (e.g. vocabulary)."""

    pad_idx: Optional[int] = None
    """The index of the pad symbol in the target domain."""

    def compute_loss(
        self, targets: Tensor, ignore_prefix_size: int = 0, label_smoothing: float = 0.0
    ) -> Tensor:
        """Compute the negative log-likelihood loss.

        :param targets:
            The target indices. *Shape:* :math:`(N,S_{tgt})`, where :math:`N` is
            the batch size and :math:`S_{tgt}` is the target sequence length.
        :param ignore_prefix_size:
            The number of logits from the beginning of the sequence that should
            be ignored in the loss computation.
        :param label_smoothing:
            The amount of label smoothing when computing the loss.
        """
        if ignore_prefix_size > 0:
            logits = self.logits[:, ignore_prefix_size:, :]
        else:
            logits = self.logits

        if ignore_prefix_size > 0:
            targets = targets[:, ignore_prefix_size:]

        # For numerical stability run in single precision.
        lprobs = log_softmax(logits, dim=-1, dtype=torch.float32)

        return nll_loss(lprobs, targets, self.pad_idx, label_smoothing)
