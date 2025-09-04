# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import final

import editdistance
import torch
from torch import Tensor
from torcheval.metrics import Metric
from typing_extensions import Self, override

from fairseq2.device import Device
from fairseq2.nn import BatchLayout
from fairseq2.utils.tensor import to_tensor


@final
class WerMetric(Metric[tuple[Tensor, Tensor]]):
    """Computes the WER (Word Error Rate)."""

    def __init__(self, *, device: Device | None = None) -> None:
        super().__init__(device=device)

        dtype = torch.int64

        unit_err = torch.zeros((), device=device, dtype=dtype)
        unit_len = torch.zeros((), device=device, dtype=dtype)

        self.unit_err: Tensor
        self.unit_len: Tensor

        self._add_state("unit_err", unit_err)
        self._add_state("unit_len", unit_len)

        word_err = torch.zeros((), device=device, dtype=dtype)
        word_len = torch.zeros((), device=device, dtype=dtype)

        self.word_err: Tensor
        self.word_len: Tensor

        self._add_state("word_err", word_err)
        self._add_state("word_len", word_len)

    @override
    @torch.inference_mode()
    def update(
        self,
        refs: Sequence[str],
        ref_seqs: Tensor,
        ref_seqs_layout: BatchLayout,
        hyps: Sequence[str],
        hyp_seqs: Tensor,
        hyp_seqs_layout: BatchLayout,
    ) -> Self:
        """
        :param refs:
            The reference strings.
        :param ref_seqs:
            The reference sequences. *Shape:* :math:`(N,S_{ref})`, where
            :math:`N` is the batch size and :math:`S_{ref}` is the sequence
            length of the references.
        :param hyps:
            The hypothesis strings.
        :param hyp_seqs:
            The hypothesis sequences. *Shape:* :math:`(N,S_{hyp})`, where
            :math:`N` is the batch size and :math:`S_{hyp}` is the sequence
            length of the hypotheses.
        """
        if ref_seqs_layout.packed or hyp_seqs_layout.packed:
            raise ValueError("`refs` and `hyps` must not be packed batches.")

        ref_seq_lens = ref_seqs_layout.seq_lens
        hyp_seq_lens = hyp_seqs_layout.seq_lens

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
    def compute(self) -> tuple[Tensor, Tensor]:
        if self.unit_len and self.word_len:
            uer = self.unit_err * 100.0 / self.unit_len
            wer = self.word_err * 100.0 / self.word_len
        else:
            uer = to_tensor(-1.0, dtype=torch.float32)
            wer = to_tensor(-1.0, dtype=torch.float32)

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
