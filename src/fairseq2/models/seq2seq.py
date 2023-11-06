# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from fairseq2.data import VocabularyInfo
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.nn.padding import PaddingMask


class Seq2SeqModel(Module, ABC):
    """Represents a sequence-to-sequence model."""

    target_vocab_info: VocabularyInfo

    def __init__(self, target_vocab_info: VocabularyInfo) -> None:
        """
        :param target_vocab_info:
            The vocabulary information of sequences produced by the model.
        """
        super().__init__()

        self.target_vocab_info = target_vocab_info

    @abstractmethod
    def forward(self, batch: Seq2SeqBatch) -> SequenceModelOutput:
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

    source_padding_mask: Optional[PaddingMask]
    """The padding mask of ``source_seqs``. *Shape:* :math:`(N,S_{src})`, where
    :math:`N` is the batch size and :math:`S_{src}` is the source sequence
    length."""

    target_seqs: Tensor
    """The target sequences. *Shape:* :math:`(N,S_{tgt},*)`, where :math:`N` is
    the batch size, :math:`S_{tgt}` is the target sequence length, and :math:`*`
    is any number of sequence-specific dimensions including none."""

    target_padding_mask: Optional[PaddingMask]
    """The padding mask of ``target_seqs``. *Shape:* :math:`(N,S_{tgt})`, where
    :math:`N` is the batch size and :math:`S_{tgt}` is the target sequence
    length."""

    example: Any = None
    """The data example from which this batch was constructed."""

    def as_training_input(self) -> Tuple[Seq2SeqBatch, Tensor]:
        """Return a copy of this batch for model training.

        :returns:
          - The batch with target sequences trimmed one step from the end to use
            as model input.
          - The target sequences trimmed one step from the beginning to use as
            targets in loss computation.
        """
        if (target_seq_len := self.target_seqs.size(1)) < 2:
            raise ValueError(
                f"The sequence length of `target_seqs` must be at least 2 for training, but is {target_seq_len} instead."
            )

        target_seqs = self.target_seqs[:, :-1]  # TODO: even padding for fp16?

        if self.target_padding_mask is None:
            target_padding_mask = None
        else:
            target_padding_mask = self.target_padding_mask.trim(1)

        batch = Seq2SeqBatch(
            self.source_seqs, self.source_padding_mask, target_seqs, target_padding_mask
        )

        return batch, self.target_seqs[:, 1:]

    @property
    def batch_size(self) -> int:
        """The size of the batch."""
        return self.target_seqs.size(0)

    def compute_num_source_tokens(self) -> Tensor:
        """Compute the number of source tokens in this batch."""
        if self.source_padding_mask is None:
            return torch.full(
                (), self.source_seqs.numel(), device=self.source_seqs.device
            )

        return self.source_padding_mask.seq_lens.sum()

    def compute_num_target_tokens(self) -> Tensor:
        """Compute the number of target tokens in this batch."""
        if self.target_padding_mask is None:
            return torch.full(
                (), self.target_seqs.numel(), device=self.target_seqs.device
            )

        return self.target_padding_mask.seq_lens.sum()
