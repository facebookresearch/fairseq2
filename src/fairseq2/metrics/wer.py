# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, final

import editdistance
import torch
from torch import Tensor
from torcheval.metrics import Metric
from typing_extensions import Self

from fairseq2.nn.padding import PaddingMask, get_seq_lens
from fairseq2.typing import Device, override


@final
class WerMetric(Metric[Tuple[Tensor, Tensor]]):
    """Computes the WER (Word Error Rate)."""

    unit_err: Tensor
    unit_len: Tensor
    word_err: Tensor
    word_len: Tensor

    def __init__(self, *, device: Optional[Device] = None) -> None:
        super().__init__(device=device)

        self._add_state("unit_err", torch.zeros((), device=device, dtype=torch.int64))
        self._add_state("unit_len", torch.zeros((), device=device, dtype=torch.int64))

        self._add_state("word_err", torch.zeros((), device=device, dtype=torch.int64))
        self._add_state("word_len", torch.zeros((), device=device, dtype=torch.int64))

    @override
    @torch.inference_mode()
    def update(
        self,
        refs: Sequence[str],
        ref_seqs: Tensor,
        ref_padding_mask: Optional[PaddingMask],
        hyps: Sequence[str],
        hyp_seqs: Tensor,
        hyp_padding_mask: Optional[PaddingMask],
    ) -> Self:
        """
        :param refs:
            The reference strings.
        :param ref_seqs:
            The reference sequences. *Shape:* :math:`(N,S_{ref})`, where
            :math:`N` is the batch size and :math:`S_{ref}` is the sequence
            length of the references.
        :param ref_padding_mask:
            The padding mask of ``ref_seqs``. *Shape:* Same as ``ref_seqs``.
        :param hyps:
            The hypothesis strings.
        :param hyp_seqs:
            The hypothesis sequences. *Shape:* :math:`(N,S_{hyp})`, where
            :math:`N` is the batch size and :math:`S_{hyp}` is the sequence
            length of the hypotheses.
        :param hyp_seqs:
            The padding mask of ``hyp_seqs``. *Shape:* Same as ``hyp_seqs``.
        """
        ref_seq_lens = get_seq_lens(ref_seqs, ref_padding_mask)
        hyp_seq_lens = get_seq_lens(hyp_seqs, hyp_padding_mask)

        for ref, ref_seq, ref_seq_len, hyp, hyp_seq, hyp_seq_len in zip(
            refs, ref_seqs, ref_seq_lens, hyps, hyp_seqs, hyp_seq_lens
        ):
            ref_words = ref.split()
            hyp_words = hyp.split()

            ref_units = ref_seq[:ref_seq_len]
            hyp_units = hyp_seq[:hyp_seq_len]

            unit_err = editdistance.eval(hyp_units, ref_units)
            word_err = editdistance.eval(hyp_words, ref_words)

            self.unit_err += unit_err
            self.word_err += word_err

            self.unit_len += len(ref_units)
            self.word_len += len(ref_words)

        return self

    @override
    @torch.inference_mode()
    def compute(self) -> Tuple[Tensor, Tensor]:
        if self.unit_len and self.word_len:
            uer = self.unit_err * 100.0 / self.unit_len
            wer = self.word_err * 100.0 / self.word_len
        else:
            uer = torch.zeros((), dtype=torch.float64)
            wer = torch.zeros((), dtype=torch.float64)

        return uer, wer

    @override
    @torch.inference_mode()
    def merge_state(self, metrics: Iterable[WerMetric]) -> Self:
        for metric in metrics:
            self.unit_err += metric.unit_err.to(self.device)
            self.unit_len += metric.unit_len.to(self.device)
            self.word_err += metric.word_err.to(self.device)
            self.word_len += metric.word_len.to(self.device)

        return self
