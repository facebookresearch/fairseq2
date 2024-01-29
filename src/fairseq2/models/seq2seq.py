# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, final

import torch
from torch import Tensor
from torch.nn import Module
from torcheval.metrics import Mean, Sum, Throughput

from fairseq2.data import VocabularyInfo
from fairseq2.gang import Gang
from fairseq2.metrics import MetricBag
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.nn.padding import PaddingMask
from fairseq2.typing import finaloverride


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

    def as_input_and_target(self) -> Tuple[Seq2SeqBatch, Tensor]:
        """Use this batch for model training or validation.

        :returns:
          - A new batch with the target sequences trimmed one step from the end
            to use as model input.
          - The target sequences trimmed one step from the beginning to use in
            loss computation.
        """
        if (seq_len := self.target_seqs.size(1)) < 2:
            raise ValueError(
                f"The sequence length of `target_seqs` must be at least 2 for training, but is {seq_len} instead."
            )

        target_seqs = self.target_seqs[:, :-1]

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
        """The size of the batch dimension."""
        return self.target_seqs.size(0)

    def num_source_elements(self) -> int:
        """Return the number of elements in the source sequences."""
        if self.source_padding_mask is None:
            return self.source_seqs.numel()

        return int(self.source_padding_mask.seq_lens.sum())

    def num_target_elements(self) -> int:
        """Return the number of elements in the target sequences."""
        if self.target_padding_mask is None:
            return self.target_seqs.numel()

        return int(self.target_padding_mask.seq_lens.sum())


@final
class Seq2SeqModelMetricBag(MetricBag):
    """Holds the common metrics of a sequence-to-sequence model."""

    loss: Mean
    entropy_loss: Mean
    batch_size: Mean
    elements_per_batch: Mean
    elements_per_second: Throughput
    num_source_elements: Sum
    num_target_elements: Sum

    def __init__(self, gang: Gang) -> None:
        """
        :param gang:
            The gang to sync metrics across all processes.
        """
        super().__init__(gang)

        d = gang.device

        self.register_metric("loss", Mean(device=d), persistent=False)

        self.register_metric("entropy_loss", Mean(device=d), persistent=False)

        self.register_metric("batch_size", Mean(device=d), persistent=False)

        self.register_metric("elements_per_batch", Mean(device=d), persistent=False)

        self.register_metric("elements_per_second", Throughput(device=d), persistent=False)  # fmt: skip

        self.num_source_elements = Sum(device=d)
        self.num_target_elements = Sum(device=d)

    def update_metrics(
        self,
        batches: Sequence[Seq2SeqBatch],
        losses: Sequence[Tensor],
        elapsed_time: float,
    ) -> None:
        """Update the metrics.

        :param batches:
            The batches processed by the model in the last training step.
        :param output:
            The losses generated by the model for each batch in ``batches``.
        :param elapsed_time:
            The total elapsed time to read and process ``batches``.
        """
        loss = torch.zeros((), dtype=torch.float64)

        batch_size = torch.zeros((), dtype=torch.float64)

        num_source_elements = torch.zeros((), dtype=torch.float64)
        num_target_elements = torch.zeros((), dtype=torch.float64)

        for batch, batch_loss in zip(batches, losses):
            loss += float(batch_loss)

            batch_size += batch.batch_size

            num_source_elements += batch.num_source_elements()
            num_target_elements += batch.num_target_elements() - batch.batch_size

        loss /= num_target_elements

        self.loss.update(loss, weight=num_target_elements)

        # Mainly exists for compatibility with fairseq's `nll_loss`.
        self.entropy_loss.update(loss / math.log(2), weight=num_target_elements)

        self.batch_size.update(batch_size * self.gang.size)

        self.elements_per_batch.update(num_target_elements * self.gang.size)

        self.elements_per_second.update(int(num_target_elements), elapsed_time)

        self.num_source_elements.update(num_source_elements)
        self.num_target_elements.update(num_target_elements)

    def reset_batch_metrics(self) -> None:
        """Reset the batch metrics to their initial state."""
        self.loss.reset()
        self.entropy_loss.reset()
        self.batch_size.reset()
        self.elements_per_batch.reset()
        self.elements_per_second.reset()

    @finaloverride
    def process_metric_values(self, values: Dict[str, Any]) -> None:
        values["elapsed_time"] = self.elements_per_second.elapsed_time_sec
