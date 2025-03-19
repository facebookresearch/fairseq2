# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Final, final

import torch
from sacrebleu import corpus_bleu
from sacrebleu.metrics.bleu import BLEU, MAX_NGRAM_ORDER
from torch import Tensor
from torcheval.metrics import Metric
from typing_extensions import Self, override

from fairseq2.typing import Device

DEFAULT_BLEU_TOKENIZER: Final = BLEU.TOKENIZER_DEFAULT


@final
class BleuMetric(Metric[Tensor]):
    """Computes the BLEU score."""

    tokenizer: str
    sys_len: Tensor
    ref_len: Tensor
    valid_ngrams: Tensor
    total_ngrams: Tensor

    def __init__(
        self,
        *,
        tokenizer: str = DEFAULT_BLEU_TOKENIZER,
        device: Device | None = None,
    ) -> None:
        super().__init__(device=device)

        if tokenizer not in BLEU.TOKENIZERS:
            raise UnknownBleuTokenizerError(tokenizer)

        self.tokenizer = tokenizer

        dtype = torch.int64

        sys_len = torch.zeros((), device=device, dtype=dtype)
        ref_len = torch.zeros((), device=device, dtype=dtype)

        self._add_state("sys_len", sys_len)
        self._add_state("ref_len", ref_len)

        valid_ngrams = torch.zeros((MAX_NGRAM_ORDER,), device=device, dtype=dtype)
        total_ngrams = torch.zeros((MAX_NGRAM_ORDER,), device=device, dtype=dtype)

        self._add_state("valid_ngrams", valid_ngrams)
        self._add_state("total_ngrams", total_ngrams)

    @override
    @torch.inference_mode()
    def update(self, refs: Sequence[str], hyps: Sequence[str]) -> Self:
        """
        :param refs: The references.
        :param hyps: The hypotheses.
        """
        bleu = corpus_bleu(hyps, [refs], tokenize=self.tokenizer)

        self.sys_len += bleu.sys_len
        self.ref_len += bleu.ref_len

        self.valid_ngrams += torch.tensor(bleu.counts, device=self.device)
        self.total_ngrams += torch.tensor(bleu.totals, device=self.device)

        return self

    @override
    @torch.inference_mode()
    def compute(self) -> Tensor:
        sys_len = int(self.sys_len)
        ref_len = int(self.ref_len)

        valid_ngrams = self.valid_ngrams.tolist()
        total_ngrams = self.total_ngrams.tolist()

        score_output = BLEU.compute_bleu(valid_ngrams, total_ngrams, sys_len, ref_len)

        return torch.tensor(score_output.score, device=self.device)

    @override
    @torch.inference_mode()
    def merge_state(self, metrics: Iterable[BleuMetric]) -> Self:
        for metric in metrics:
            self.sys_len += metric.sys_len.to(self.device)
            self.ref_len += metric.ref_len.to(self.device)

            self.valid_ngrams += metric.valid_ngrams.to(self.device)
            self.total_ngrams += metric.total_ngrams.to(self.device)

        return self


class UnknownBleuTokenizerError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(
            f"'{name}' is not a known BLEU tokenizer. See the documentation of the sacrebleu package."
        )

        self.name = name
