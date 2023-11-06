# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import log_softmax

from fairseq2.data import VocabularyInfo
from fairseq2.nn.functional import nll_loss
from fairseq2.nn.padding import PaddingMask


class SequenceModel(Module, ABC):
    """Represents a sequence model."""

    vocab_info: VocabularyInfo

    def __init__(self, vocab_info: VocabularyInfo) -> None:
        """
        :param vocab_info:
            The vocabulary information of sequences produced by the model.
        """
        super().__init__()

        self.vocab_info = vocab_info

    @abstractmethod
    def forward(self, batch: SequenceBatch) -> SequenceModelOutput:
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

    padding_mask: Optional[PaddingMask]
    """The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N` is
    the batch size and :math:`S` is the sequence length."""

    example: Any = None
    """The data example from which this batch was constructed."""

    @property
    def batch_size(self) -> int:
        """The size of the batch."""
        return self.seqs.size(0)

    def compute_num_tokens(self) -> Tensor:
        """Compute the number of tokens in this batch."""
        if self.padding_mask is None:
            return torch.full((), self.seqs.numel(), device=self.seqs.device)

        return self.padding_mask.seq_lens.sum()


@dataclass
class SequenceModelOutput:
    """Holds the output of a sequence model."""

    logits: Tensor
    """The logits for next-step prediction. *Shape:* :math:`(N,S,T)`, where
    :math:`N` is the batch size, :math:`S` is the sequence length, and :math:`T`
    is the size of the vocabulary."""

    vocab_info: VocabularyInfo
    """The vocabulary information."""

    def compute_loss(
        self,
        targets: Tensor,
        *,
        ignore_prefix_size: int = 0,
        label_smoothing: float = 0.0,
    ) -> Tensor:
        """Compute the negative log-likelihood loss.

        :param targets:
            The target indices. *Shape:* :math:`(N,S)`, where :math:`N` is the
            batch size and :math:`S` is the sequence length.
        :param ignore_prefix_size:
            The number of steps from the beginning of the sequence that should
            be ignored in the loss computation.
        :param label_smoothing:
            The amount of label smoothing to apply while computing the loss.
        """
        if ignore_prefix_size > 0:
            logits = self.logits[:, ignore_prefix_size:, :]
        else:
            logits = self.logits

        if ignore_prefix_size > 0:
            targets = targets[:, ignore_prefix_size:]

        # For numerical stability run in single precision.
        lprobs = log_softmax(logits, dim=-1, dtype=torch.float32)

        return nll_loss(
            lprobs, targets, self.vocab_info.pad_idx, label_smoothing=label_smoothing
        )
