# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import ctc_loss, log_softmax

from fairseq2.nn import BatchLayout
from fairseq2.nn.utils.padding import pad_seqs


class AsrModel(Module, ABC):
    """Represents an Automatic Speech Recognition model."""

    @abstractmethod
    def forward(self, seqs: Tensor, seqs_layout: BatchLayout) -> AsrModelOutput: ...


@dataclass
class AsrModelOutput:
    logits: Tensor
    """The logits for next-step prediction. *Shape:* :math:`(N,S_{out},T)`,
    where :math:`N` is the batch size, :math:`S_{out}` is the output sequence
    length, and :math:`T` is the size of the vocabulary."""

    logits_layout: BatchLayout

    def __post_init__(self) -> None:
        if self.logits_layout.packed:
            raise ValueError("`logits` must not be a packed batch.")

    def compute_loss(self, targets: Tensor, targets_layout: BatchLayout) -> Tensor:
        """Compute the CTC (Connectionist Temporal Classification) loss.

        :param targets:
            The target indices. *Shape:* :math:`(N,S_{tgt})`, where :math:`N` is
            the batch size and :math:`S_{tgt}` is the target sequence length.
        :param target_padding_mask:
            The padding mask of the targets. *Shape:* Same as ``targets``.

        :returns:
            A scalar tensor representing the summed CTC loss.
        """
        if targets_layout.packed:
            raise ValueError("`targets` must not be a packed batch.")

        # For numerical stability run in single precision.
        # (N, S, T)
        lprobs = log_softmax(self.logits, dim=-1, dtype=torch.float32)

        # (N, S, T) -> (S, N, T)
        lprobs_t = lprobs.transpose(0, 1)

        # ()
        return ctc_loss(
            log_probs=lprobs_t,
            input_lengths=self.logits_layout.seq_lens_pt,
            targets=targets,
            target_lengths=targets_layout.seq_lens_pt,
            reduction="sum",
            zero_infinity=True,
        )

    def generate_hypotheses(
        self, pad_idx: int, blank_label: int = 0
    ) -> tuple[Tensor, BatchLayout]:
        """Generate hypotheses using greedy search.

        :param pad_idx:
            The index of the PAD symbol in the target vocabulary.
        :param blank_label:
            The blank label in logits.

        :returns:
            - The generated token (i.e. unit) sequences. *Shape:* :math:`(N,S)`,
              where :math:`N` is the batch size and :math:`S` is the sequence
              length.
            - The padding mask of the generated sequences. *Shape:* Same as the
              generated sequences.
        """
        hyp_seqs = []

        # Get the greedy token (i.e. unit) output of the model.
        for logits, logits_len in zip(self.logits, self.logits_layout.seq_lens):
            # (S)
            hyp_seq = logits[:logits_len].argmax(-1).unique_consecutive()

            # (S - blank)
            hyp_seq = hyp_seq[hyp_seq != blank_label]

            hyp_seqs.append(hyp_seq)

        # (N, S), (N, S)
        return pad_seqs(hyp_seqs, pad_value=pad_idx)
