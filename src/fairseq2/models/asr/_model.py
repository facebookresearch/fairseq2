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

from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import PaddingMask, get_seq_lens, pad_seqs


class AsrModel(Module, ABC):
    """Represents an Automatic Speech Recognition model."""

    @abstractmethod
    def forward(self, batch: SequenceBatch) -> AsrModelOutput: ...


@dataclass
class AsrModelOutput:
    logits: Tensor
    """The logits for next-step prediction. *Shape:* :math:`(N,S_{out},T)`,
    where :math:`N` is the batch size, :math:`S_{out}` is the output sequence
    length, and :math:`T` is the size of the vocabulary."""

    padding_mask: PaddingMask | None
    """The padding mask of :attr:`logits`. *Shape:* :math:`(N,S_{out})`, where
    :math:`N` is the batch size and :math:`S_{out}` is the output sequence
    length."""

    def compute_loss(
        self, targets: Tensor, target_padding_mask: PaddingMask | None
    ) -> Tensor:
        """Compute the CTC (Connectionist Temporal Classification) loss.

        :param targets:
            The target indices. *Shape:* :math:`(N,S_{tgt})`, where :math:`N` is
            the batch size and :math:`S_{tgt}` is the target sequence length.
        :param target_padding_mask:
            The padding mask of the targets. *Shape:* Same as ``targets``.

        :returns:
            A scalar tensor representing the summed CTC loss.
        """
        # For numerical stability run in single precision.
        # (N, S, T)
        lprobs = log_softmax(self.logits, dim=-1, dtype=torch.float32)

        # (N, S, T) -> (S, N, T)
        lprobs_t = lprobs.transpose(0, 1)

        # (N)
        seq_lens = get_seq_lens(lprobs, self.padding_mask)

        # (N)
        target_seq_lens = get_seq_lens(targets, target_padding_mask)

        # ()
        return ctc_loss(
            lprobs_t,
            targets,
            seq_lens,
            target_seq_lens,
            reduction="sum",
            zero_infinity=True,
        )

    def generate_hypotheses(
        self, pad_idx: int, blank_label: int = 0
    ) -> tuple[Tensor, PaddingMask | None]:
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
        seq_lens = get_seq_lens(self.logits, self.padding_mask)

        hyp_seq_list = []

        # Get the greedy token (i.e. unit) output of the model.
        for logits, seq_len in zip(self.logits, seq_lens):
            # (S)
            hyp_seq = logits[:seq_len].argmax(-1).unique_consecutive()

            # (S - blank)
            hyp_seq = hyp_seq[hyp_seq != blank_label]

            hyp_seq_list.append(hyp_seq)

        # (N, S), (N, S)
        return pad_seqs(hyp_seq_list, pad_value=pad_idx)
