# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, final

from torch import Tensor
from torch.nn import Module

from fairseq2.nn import BatchLayout
from fairseq2.nn.ops import CrossEntropy, cross_entropy


class SequenceModel(Module, ABC):
    """Represents a sequence model."""

    max_seq_len: int

    def __init__(self, max_seq_len: int) -> None:
        """
        :param max_seq_len: The maximum length of produced sequences.
        """
        super().__init__()

        self.max_seq_len = max_seq_len

    @abstractmethod
    def forward(
        self, seqs: Tensor, seqs_layout: BatchLayout
    ) -> SequenceModelOutput: ...


@final
@dataclass
class SequenceModelOutput:
    """Holds the output of a sequence model."""

    logits: Tensor
    """The logits for next-step prediction. *Shape:* :math:`(N,S,T)`, where
    :math:`N` is the batch size, :math:`S` is the sequence length, and :math:`T`
    is the size of the vocabulary."""

    pad_idx: int | None
    """The index of the PAD symbols in the vocabulary."""

    loss_fn: CrossEntropy = field(default=cross_entropy)

    def compute_loss(
        self,
        targets: Tensor,
        *,
        loss_mask: Tensor | None = None,
        reduction: Literal["sum", "mean"] = "sum",
        label_smoothing: float = 0.0,
        ignore_prefix_size: int = 0,
    ) -> Tensor:
        """
        Computes the negative log-likelihood loss.

        :param targets: The target indices. *Shape:* :math:`(N,S)`, where
            :math:`N` is the batch size and :math:`S` is the sequence length.
        :param loss_mask: The loss mask that specifies the elements in ``targets``
            that should be used in the loss computation. All non-masked elements
            will be ignored. *Shape:* Same as ``targets``.
        :param label_smoothing: The amount of label smoothing to apply while
            computing the loss.
        :param ignore_prefix_size: The number of steps from the beginning of the
            sequence that should be ignored in the loss computation.

        :returns: A scalar tensor representing the loss.
        """
        if ignore_prefix_size > 0:
            logits = self.logits[..., ignore_prefix_size:, :]
        else:
            logits = self.logits

        if ignore_prefix_size > 0:
            targets = targets[..., ignore_prefix_size:]

        if logits.ndim == 3:
            # (N, S, T) -> (N x S, T)
            logits = logits.flatten(0, 1)

        if targets.ndim == 2:
            # (N, S) -> (N x S)
            targets = targets.flatten(0, 1)

        # sum/mean: (), none: (N x S)
        loss = self.loss_fn(
            logits,
            targets,
            pad_idx=self.pad_idx,
            label_smoothing=label_smoothing,
            reduction=reduction if loss_mask is None else "none",
        )

        if loss_mask is None:
            return loss

        if ignore_prefix_size > 0:
            loss_mask = loss_mask[..., ignore_prefix_size:]

        if loss_mask.ndim == 2:
            # (N, S) -> (N x S)
            loss_mask = loss_mask.flatten(0, 1)

        loss = loss * loss_mask

        if reduction == "sum":
            return loss.sum()

        if reduction == "mean":
            return loss.mean()

        raise ValueError(
            f"`reduction` must be 'sum' or 'mean', but is '{reduction}' instead."
        )
