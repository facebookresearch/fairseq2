# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, final

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
from fairseq2.typing import override
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
@dataclass
class SequenceBatch:
    """Represents a sequence batch."""

    seqs: Tensor
    """The sequences. *Shape:* :math:`(N,S,*)`, where :math:`N` is the batch
    size, :math:`S` is the sequence length, and :math:`*` is any number of
    sequence-specific dimensions including none."""

    padding_mask: Optional[PaddingMask]
    """The padding mask of :attr:`seqs`. *Shape:* :math:`(N,S)`, where :math:`N`
    is the batch size and :math:`S` is the sequence length."""

    target_mask: Optional[Tensor] = None
    """The mask specifying the elements in ``seqs`` that should be treated as
    targets during model training or validation. *Shape:* :math:`(N,S)`, where
    :math:`N` is the batch size and :math:`S` is the sequence length."""

    example: Any = None
    """The data example from which this batch was constructed."""

    @property
    def batch_size(self) -> int:
        """The size of the batch dimension."""
        return self.seqs.size(0)

    def num_elements(self) -> int:
        """Return the number of elements in the batch."""
        if self.padding_mask is None:
            return self.seqs.numel()

        return int(self.padding_mask.seq_lens.sum())

    def num_target_elements(self) -> int:
        """Return the number of target elements in the batch."""
        if self.target_mask is not None:
            return int(self.target_mask.sum())

        return self.num_elements()


def as_auto_regressive_input(
    batch: SequenceBatch,
) -> Tuple[SequenceBatch, SequenceBatch]:
    """Use ``batch`` to train an auto-regressive model.

    :returns:
        The tuple of input and target batches.
    """
    if (seq_len := batch.seqs.size(1)) < 2:
        raise ValueError(
            f"The sequence length of `batch.seqs` must be at least 2 for training, but is {seq_len} instead."
        )

    seqs, targets = batch.seqs[:, :-1], batch.seqs[:, 1:]

    if batch.padding_mask is None:
        padding_mask = None
    else:
        padding_mask = batch.padding_mask.trim(1)

    if batch.target_mask is None:
        seqs_target_mask, target_mask = None, None
    else:
        seqs_target_mask, target_mask = (
            batch.target_mask[:, :-1], batch.target_mask[:, 1:]  # fmt: skip
        )

    batch = SequenceBatch(seqs, padding_mask, seqs_target_mask, batch.example)

    target_batch = SequenceBatch(targets, padding_mask, target_mask)

    return batch, target_batch


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
        loss_mask: Optional[Tensor] = None,
        ignore_prefix_size: int = 0,
        label_smoothing: float = 0.0,
    ) -> Tensor:
        """Compute the NLL (negative log-likelihood) loss.

        :param targets:
            The target indices. *Shape:* :math:`(N,S)`, where :math:`N` is the
            batch size and :math:`S` is the sequence length.
        :param loss_mask:
            The loss mask that specifies the elements in ``targets`` that should
            be used in the loss computation. All non-masked elements will be
            ignored. *Shape:* Same as ``targets``.
        :param ignore_prefix_size:
            The number of steps from the beginning of the sequence that should
            be ignored in the loss computation.
        :param label_smoothing:
            The amount of label smoothing to apply while computing the loss.

        :returns:
            A scalar tensor representing the summed NLL loss.
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

        # sum: (), none: (N, S)
        loss = nll_loss(
            lprobs,
            targets,
            self.pad_idx,
            label_smoothing=label_smoothing,
            reduction="sum" if loss_mask is None else "none",
        )

        if loss_mask is None:
            return loss

        if ignore_prefix_size > 0:
            loss_mask = loss_mask[:, ignore_prefix_size:]

        # ()
        return (loss * loss_mask).sum()


class SequenceModelMetricBag(MetricBag):
    """Holds the common metrics of a sequence model training."""

    nll_loss: Mean
    batch_size: Mean
    gradient_norm: Mean
    elements_per_batch: Mean
    elements_per_second: Throughput
    num_examples: Sum
    num_elements: Sum
    num_target_elements: Sum

    def __init__(self, gang: Gang, *, wall_time: Optional[Stopwatch] = None) -> None:
        """
        :param gang:
            The gang over which to sync metrics.
        :param wall_time:
            The :class:`Stopwatch` to keep track of process wall time.
        """
        super().__init__(gang, wall_time)

        d = gang.device

        self.register_metric("nll_loss", Mean(device=d), persistent=False)

        self.register_metric("batch_size", Mean(device=d), persistent=False)

        self.register_metric("gradient_norm", Mean(device=d), persistent=False)

        self.register_metric("elements_per_batch", Mean(device=d), persistent=False)

        self.register_metric(
            "elements_per_second", Throughput(device=d), persistent=False
        )

        self.num_examples = Sum(device=d)

        self.num_elements = Sum(device=d)
        self.num_target_elements = Sum(device=d)

    @torch.inference_mode()
    def update_step_metrics(
        self,
        batches: Sequence[SequenceBatch],
        nll_loss: Tensor,
        time: Stopwatch,
        gradient_norm: Optional[Tensor] = None,
    ) -> None:
        """Update the step metrics.

        :param batches:
            The batches processed by the model.
        :param nll_loss:
            The total NLL loss generated by the model for ``batches``.
        :param time:
            :class:`Stopwatch` to report elapsed time.
        :param gradient_norm:
            The total model gradient norm after backpropagating ``batches``.
        """
        batch_size = torch.zeros((), dtype=torch.float64)

        num_elements = torch.zeros((), dtype=torch.float64)
        num_target_elements = torch.zeros((), dtype=torch.float64)

        for batch in batches:
            batch_size += batch.batch_size

            num_elements += batch.num_elements()
            num_target_elements += batch.num_target_elements()

        normalized_nll_loss = nll_loss.cpu() / num_target_elements

        self.nll_loss.update(normalized_nll_loss, weight=num_target_elements)

        self.batch_size.update(batch_size * self._gang.size)

        if gradient_norm:
            self.gradient_norm.update(gradient_norm)

        self.elements_per_batch.update(num_elements * self._gang.size)

        self.elements_per_second.update(int(num_elements), time.get_elapsed_time())

        self.num_examples.update(batch_size)

        self.num_elements.update(num_elements)
        self.num_target_elements.update(num_target_elements)

    def reset_step_metrics(self) -> None:
        """Reset the step metrics to their initial state."""
        self.nll_loss.reset()
        self.batch_size.reset()
        self.gradient_norm.reset()
        self.elements_per_batch.reset()
        self.elements_per_second.reset()

    @override
    def process_metric_values(self, values: Dict[str, Any]) -> None:
        super().process_metric_values(values)

        if values["gradient_norm"] == 0.0:
            del values["gradient_norm"]

        values["elapsed_time"] = self.elements_per_second.elapsed_time_sec
