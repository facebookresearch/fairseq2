# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Iterable, Optional, TextIO, Tuple, final

import editdistance
import torch
from torch import Tensor
from torcheval.metrics import Metric
from typing_extensions import Self

from fairseq2.data.text import TextTokenDecoder
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

        self._add_state("unit_err", torch.zeros((), device=device, dtype=torch.float64))
        self._add_state("unit_len", torch.zeros((), device=device, dtype=torch.float64))

        self._add_state("word_err", torch.zeros((), device=device, dtype=torch.float64))
        self._add_state("word_len", torch.zeros((), device=device, dtype=torch.float64))

    @override
    @torch.inference_mode()
    def update(
        self,
        text_decoder: TextTokenDecoder,
        ref_seqs: Tensor,
        ref_padding_mask: Optional[PaddingMask],
        hyp_seqs: Tensor,
        hyp_padding_mask: Optional[PaddingMask],
        *,
        output_fp: Optional[TextIO] = None,
    ) -> Self:
        """
        :param text_decoder:
            The text token decoder to use.
        :param ref_seqs:
            The reference sequences. *Shape:* :math:`(N,S_{ref})`, where
            :math:`N` is the batch size and :math:`S_{ref}` is the sequence
            length of the references.
        :param ref_padding_mask:
            The padding mask of ``ref_seqs``. *Shape:* Same as ``ref_seqs``.
        :param hyp_seqs:
            The hypotheses generated by the model. *Shape:* :math:`(N,S_{hyp})`,
            where :math:`N` is the batch size and :math:`S_{hyp}` is the
            sequence length of the hypotheses.
        :param hyp_seqs:
            The padding mask of ``hyp_seqs``. *Shape:* Same as ``hyp_seqs``.
        :param output_fp:
            The file descriptor to dump the transcriptions, WER, and UER metrics.
        """
        ref_seq_lens = get_seq_lens(ref_seqs, ref_padding_mask)
        hyp_seq_lens = get_seq_lens(hyp_seqs, hyp_padding_mask)

        for ref_seq, ref_seq_len, hyp_seq, hyp_seq_len in zip(
            ref_seqs, ref_seq_lens, hyp_seqs, hyp_seq_lens
        ):
            ref_units = ref_seq[:ref_seq_len]
            hyp_units = hyp_seq[:hyp_seq_len]

            ref_text = text_decoder(ref_units)
            hyp_text = text_decoder(hyp_units)

            ref_words = ref_text.split()
            hyp_words = hyp_text.split()

            unit_err = editdistance.eval(hyp_units, ref_units)
            word_err = editdistance.eval(hyp_words, ref_words)

            self.unit_err += unit_err
            self.word_err += word_err

            self.unit_len += len(ref_units)
            self.word_len += len(ref_words)

            if output_fp is not None:
                output_fp.write(f"REF: {ref_text}\n")
                output_fp.write(f"HYP: {hyp_text}\n")

                output_fp.write(f"UNIT ERROR: {unit_err}\n")
                output_fp.write(f"WORD ERROR: {word_err}\n")

                output_fp.write("\n")

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
