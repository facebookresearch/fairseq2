# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import log_softmax

from fairseq2.nn.functional import nll_loss


class SequenceModel(Module, ABC):
    """Represents a sequence model."""

    @abstractmethod
    def forward(self, batch: "SequenceBatch") -> "SequenceModelOutput":
        """
        :param batch:
            The batch of sequences to process.
        """


@dataclass
class SequenceBatch:
    """Represents a sequence batch."""

    seqs: Tensor
    """The sequences. *Shape:* :math:`(N,S,*)`, where :math:`N` is the batch
    size, :math:`S` is the sequence length, and :math:`*` is any number of
    sequence-specific dimensions including none."""

    seq_lens: Optional[Tensor]
    """An array where each element represents the length of the sequence at the
    same index in :attr:`seqs`. *Shape:* :math:`(N)`, where :math:`N` is the
    batch size."""

    example: Any = None
    """The data example from which this batch was constructed."""

    @property
    def batch_size(self) -> int:
        """The size of the batch."""
        return self.seqs.size(0)

    def num_tokens(self) -> Tensor:
        """Return the number of tokens."""
        if self.seq_lens is None:
            return torch.full((), self.seqs.numel(), device=self.seqs.device)

        return self.seq_lens.sum()


@dataclass
class SequenceModelOutput:
    """Holds the output of a sequence model."""

    logits: Tensor
    """The logits for next-step prediction. *Shape:* :math:`(N,S,T)`, where
    :math:`N` is the batch size, :math:`S` is the sequence length, and :math:`T`
    is the size of the target vocabulary."""

    pad_idx: Optional[int] = None
    """The index of the pad symbol in the target vocabulary."""

    def compute_loss(
        self, targets: Tensor, ignore_prefix_size: int = 0, label_smoothing: float = 0.0
    ) -> Tensor:
        """Compute the negative log-likelihood loss.

        :param targets:
            The target indices. *Shape:* :math:`(N,S)`, where :math:`N` is the
            batch size and :math:`S` is the sequence length.
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
