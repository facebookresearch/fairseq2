# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Iterable, Optional, Sequence, final

import torch
from sacrebleu import corpus_bleu
from sacrebleu.metrics.bleu import BLEU, MAX_NGRAM_ORDER
from torch import Tensor
from torcheval.metrics import Metric
from typing_extensions import Self

from fairseq2.typing import Device, override


@final
class BleuMetric(Metric[Tensor]):
    """Computes the BLEU score."""

    sys_len: Tensor
    ref_len: Tensor
    valid_ngrams: Tensor
    total_ngrams: Tensor

    def __init__(self, *, device: Optional[Device] = None) -> None:
        super().__init__(device=device)

        self._add_state("sys_len", torch.zeros((), device=device, dtype=torch.int64))
        self._add_state("ref_len", torch.zeros((), device=device, dtype=torch.int64))

        self._add_state("valid_ngrams", torch.zeros((MAX_NGRAM_ORDER,), device=device, dtype=torch.int64))  # fmt: skip
        self._add_state("total_ngrams", torch.zeros((MAX_NGRAM_ORDER,), device=device, dtype=torch.int64))  # fmt: skip

    @override
    @torch.inference_mode()
    def update(self, refs: Sequence[str], hyps: Sequence[str]) -> Self:
        """
        :param refs:
            The reference strings.
        :param hyps:
            The hypothesis strings.
        """
        device = self.sys_len.device

        bleu = corpus_bleu(hyps, [refs])

        self.sys_len += bleu.sys_len
        self.ref_len += bleu.ref_len

        self.valid_ngrams += torch.tensor(bleu.counts, device=device)
        self.total_ngrams += torch.tensor(bleu.totals, device=device)

        return self

    @override
    @torch.inference_mode()
    def compute(self) -> Tensor:
        valid_ngrams = self.valid_ngrams.tolist()
        total_ngrams = self.total_ngrams.tolist()

        bleu = BLEU.compute_bleu(
            valid_ngrams, total_ngrams, int(self.sys_len), int(self.ref_len)
        )

        return torch.tensor(bleu.score, device=self.sys_len.device)

    @override
    @torch.inference_mode()
    def merge_state(self, metrics: Iterable[BleuMetric]) -> Self:
        for metric in metrics:
            self.sys_len += metric.sys_len.to(self.device)
            self.ref_len += metric.ref_len.to(self.device)

            self.valid_ngrams += metric.valid_ngrams.to(self.device)
            self.total_ngrams += metric.total_ngrams.to(self.device)

        return self
