# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.device import SupportsDeviceTransfer
from fairseq2.models.sequence import SequenceBatch, SequenceModelOutput
from fairseq2.nn.padding import PaddingMask
from fairseq2.typing import Device


class Seq2SeqModel(Module, ABC):
    """Represents a sequence-to-sequence model."""

    max_source_seq_len: int
    max_target_seq_len: int

    def __init__(self, max_source_seq_len: int, max_target_seq_len: int) -> None:
        """
        :param max_target_seq_len: The maximum length of produced sequences.
        """
        super().__init__()

        self.max_source_seq_len = max_source_seq_len
        self.max_target_seq_len = max_target_seq_len

    @abstractmethod
    def forward(self, batch: Seq2SeqBatch) -> SequenceModelOutput:
        """
        :param batch: The batch of sequences to process.
        """


@dataclass
class Seq2SeqBatch(SupportsDeviceTransfer):
    """Represents a sequence-to-sequence batch."""

    source_seqs: Tensor
    """The source sequences. *Shape:* :math:`(N,S_{src},*)`, where :math:`N` is
    the batch size, :math:`S_{src}` is the source sequence length, and :math:`*`
    is any number of sequence-specific dimensions including none."""

    source_padding_mask: PaddingMask | None
    """The padding mask of :attr:`source_seqs`. *Shape:* :math:`(N,S_{src})`,
    where :math:`N` is the batch size and :math:`S_{src}` is the source sequence
    length."""

    target_seqs: Tensor
    """The target sequences. *Shape:* :math:`(N,S_{tgt},*)`, where :math:`N` is
    the batch size, :math:`S_{tgt}` is the target sequence length, and :math:`*`
    is any number of sequence-specific dimensions including none."""

    target_padding_mask: PaddingMask | None
    """The padding mask of :attr:`target_seqs`. *Shape:* :math:`(N,S_{tgt})`,
    where :math:`N` is the batch size and :math:`S_{tgt}` is the target sequence
    length."""

    example: object = None
    """The data example from which this batch was constructed."""

    @property
    def batch_size(self) -> int:
        """The size of the batch dimension."""
        return self.target_seqs.size(0)

    def num_source_elements(self) -> int:
        """Return the number of source elements in the batch."""
        if self.source_padding_mask is None:
            return self.source_seqs.numel()

        return int(self.source_padding_mask.seq_lens.sum())

    def num_target_elements(self) -> int:
        """Return the number of target elements in the batch."""
        if self.target_padding_mask is None:
            return self.target_seqs.numel()

        return int(self.target_padding_mask.seq_lens.sum())

    @override
    def to(self, device: Device) -> None:
        self.source_seqs = self.source_seqs.to(device)

        if self.source_padding_mask is not None:
            self.source_padding_mask = self.source_padding_mask.to(device)

        self.target_seqs = self.target_seqs.to(device)

        if self.target_padding_mask is not None:
            self.target_padding_mask = self.target_padding_mask.to(device)


def as_auto_regressive_input(batch: Seq2SeqBatch) -> tuple[Seq2SeqBatch, SequenceBatch]:
    """Use ``batch`` to train an auto-regressive model.

    :returns:
        The tuple of input and target batches.
    """
    if (seq_len := batch.target_seqs.size(1)) < 2:
        raise ValueError(
            f"The sequence length of `batch.target_seqs` must be at least 2 for training, but is {seq_len} instead."
        )

    seqs, targets = batch.target_seqs[:, :-1], batch.target_seqs[:, 1:]

    if batch.target_padding_mask is None:
        padding_mask = None
    else:
        padding_mask = batch.target_padding_mask.trim(1)

    batch = Seq2SeqBatch(
        batch.source_seqs,
        batch.source_padding_mask,
        seqs,
        padding_mask,
        batch.example,
    )

    target_batch = SequenceBatch(targets, padding_mask)

    return batch, target_batch
