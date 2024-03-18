# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, final

import torch
from torch import Tensor
from torch.nn.functional import log_softmax
from torcheval.metrics import Mean, Sum, Throughput

from fairseq2.data import VocabularyInfo
from fairseq2.gang import Gang
from fairseq2.metrics import MetricBag
from fairseq2.models.model import Model
from fairseq2.nn.functional import nll_loss
from fairseq2.nn.padding import PaddingMask
from fairseq2.utils.profiler import Stopwatch


class SequenceModel(Model, ABC):
    """Represents a sequence model."""

    max_seq_len: int
    vocab_info: VocabularyInfo

    def __init__(self, max_seq_len: int, vocab_info: VocabularyInfo) -> None:
        """
        :param max_seq_len:
            The maximum length of sequences produced by the model.
        :param vocab_info:
            The vocabulary information of sequences produced by the model.
        """
        super().__init__()

        self.max_seq_len = max_seq_len
        self.vocab_info = vocab_info

    @abstractmethod
    def forward(self, batch: SequenceBatch) -> SequenceModelOutput:
        """
        :param batch:
            The batch of sequences to process.
        """


@final
@dataclass(frozen=True)
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

    def num_elements(self) -> int:
        """Returns the number of elements in the sequences."""
        if self.padding_mask is None:
            return self.seqs.numel()

        return int(self.padding_mask.seq_lens.sum())


# compat
@dataclass
class BCVocabInfo:
    pad_idx: Optional[int] = None


@final
@dataclass
class SequenceModelOutput:
    """Holds the output of a sequence model."""

    logits: Tensor
    """The logits for next-step prediction. *Shape:* :math:`(N,S,T)`, where
    :math:`N` is the batch size, :math:`S` is the sequence length, and :math:`T`
    is the size of the vocabulary."""

    pad_idx: Optional[int]
    """The index of the PAD symbols in the vocabulary."""

    # compat
    vocab_info: BCVocabInfo = field(default_factory=BCVocabInfo)

    # compat
    def __post_init__(self) -> None:
        self.vocab_info.pad_idx = self.pad_idx

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
        # (N, S, T)
        lprobs = log_softmax(logits, dim=-1, dtype=torch.float32)

        # ()
        return nll_loss(lprobs, targets, self.pad_idx, label_smoothing=label_smoothing)


@final
class SequenceModelMetricBag(MetricBag):
    """Holds the common metrics of a sequence model."""

    nll_loss: Mean
    batch_size: Mean
    elements_per_batch: Mean
    elements_per_second: Throughput
    num_examples: Sum
    num_elements: Sum

    def __init__(self, gang: Gang, wall_time: Optional[Stopwatch] = None) -> None:
        """
        :param gang:
            The gang to sync metrics across all processes.
        :param wall_time:
            The :class:`Stopwatch` to keep track of process wall time.
        """
        super().__init__(gang, wall_time)

        d = gang.device

        self.register_metric("nll_loss", Mean(device=d), persistent=False)

        self.register_metric("batch_size", Mean(device=d), persistent=False)

        self.register_metric("elements_per_batch", Mean(device=d), persistent=False)

        self.register_metric(
            "elements_per_second", Throughput(device=d), persistent=False
        )

        self.num_examples = Sum(device=d)
        self.num_elements = Sum(device=d)

    def update_metrics(
        self,
        batches: Sequence[SequenceBatch],
        nll_losses: Sequence[Tensor],
        elapsed_time: float,
    ) -> None:
        """Update the metrics.

        :param batches:
            The batches processed by the model in the last training step.
        :param nll_losses:
            The NLL losses generated by the model for each batch in ``batches``.
        :param elapsed_time:
            The total elapsed time to read and process ``batches``.
        """
        nll_loss = torch.zeros((), dtype=torch.float64)

        batch_size = torch.zeros((), dtype=torch.float64)

        num_elements = torch.zeros((), dtype=torch.float64)

        for batch, batch_nll_loss in zip(batches, nll_losses):
            nll_loss += float(batch_nll_loss)

            batch_size += batch.batch_size

            num_elements += batch.num_elements()

        nll_loss /= num_elements

        self.nll_loss.update(nll_loss, weight=num_elements)

        self.batch_size.update(batch_size * self._gang.size)

        self.elements_per_batch.update(num_elements * self._gang.size)

        self.elements_per_second.update(int(num_elements), elapsed_time)

        self.num_examples.update(batch_size)

        self.num_elements.update(num_elements)

    def reset_batch_metrics(self) -> None:
        """Reset the batch metrics to their initial state."""
        self.nll_loss.reset()
        self.batch_size.reset()
        self.elements_per_batch.reset()
        self.elements_per_second.reset()
