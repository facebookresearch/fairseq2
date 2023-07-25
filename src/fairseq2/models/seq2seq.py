# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from fairseq2.models.sequence import SequenceModelOutput


class Seq2SeqModel(Module, ABC):
    """Represents a sequence-to-sequence model."""

    @abstractmethod
    def forward(self, batch: "Seq2SeqBatch") -> SequenceModelOutput:
        """
        :param batch:
            The batch of sequences to process.
        """


@dataclass
class Seq2SeqBatch:
    """Represents a sequence-to-sequence batch."""

    source_seqs: Tensor
    """The source sequences. *Shape:* :math:`(N,S_{src},*)`, where :math:`N` is
    the batch size, :math:`S_{src}` is the source sequence length, and :math:`*`
    is any number of sequence-specific dimensions including none."""

    source_seq_lens: Optional[Tensor]
    """An array where each element represents the length of the sequence at the
    same index in :attr:`source_seqs`. *Shape:* :math:`(N)`, where :math:`N` is
    the batch size."""

    target_seqs: Tensor
    """The target sequences. *Shape:* :math:`(N,S_{tgt},*)`, where :math:`N` is
    the batch size, :math:`S_{tgt}` is the target sequence length, and :math:`*`
    is any number of sequence-specific dimensions including none."""

    target_seq_lens: Optional[Tensor]
    """An array where each element represents the length of the sequence at the
    same index in :attr:`target_seqs`. *Shape:* :math:`(N)`, where :math:`N` is
    the batch size."""

    example: Any = None
    """The data example from which this batch was constructed."""

    def as_training_input(self) -> Tuple["Seq2SeqBatch", Tensor]:
        """Return a copy of this batch for model training.

        :returns:
          - The copy with target sequences trimmed one step from the end.
          - The target sequences shifted one step to the left for use as targets
            in loss computation.
        """
        target_seqs = self.target_seqs[:, :-1]  # TODO: even padding for fp16?

        if self.target_seq_lens is None:
            target_seq_lens = None
        else:
            target_seq_lens = self.target_seq_lens - 1

        batch = Seq2SeqBatch(
            self.source_seqs, self.source_seq_lens, target_seqs, target_seq_lens
        )

        return batch, self.target_seqs[:, 1:]

    @property
    def batch_size(self) -> int:
        """The size of the batch."""
        return self.target_seqs.size(0)

    def num_source_tokens(self) -> Tensor:
        """Return the number of source tokens."""
        if self.source_seq_lens is None:
            return torch.full(
                (), self.source_seqs.numel(), device=self.source_seqs.device
            )

        return self.source_seq_lens.sum()

    def num_target_tokens(self) -> Tensor:
        """Return the number of target tokens."""
        if self.target_seq_lens is None:
            return torch.full(
                (), self.target_seqs.numel(), device=self.target_seqs.device
            )

        return self.target_seq_lens.sum()
