# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Final, Iterable, Optional, Sequence, final

import torch
from sacrebleu.metrics.chrf import CHRF
from torch import Tensor
from torcheval.metrics import Metric
from typing_extensions import Self

from fairseq2.typing import Device, override


@final
class ChrfMetric(Metric[Tensor]):
    """Computes the chrF++ score."""

    CHAR_ORDER: Final = 6
    WORD_ORDER: Final = 2  # chrF++

    stats: Tensor

    def __init__(self, *, device: Optional[Device] = None) -> None:
        super().__init__(device=device)

        stats_len = 3 * (self.CHAR_ORDER + self.WORD_ORDER)

        stats = torch.zeros((stats_len,), device=device, dtype=torch.int64)

        self._add_state("stats", stats)

    @override
    @torch.inference_mode()
    def update(self, refs: Sequence[str], hyps: Sequence[str]) -> Self:
        """
        :param refs:
            The references.
        :param hyps:
            The hypotheses.
        """
        chrf = self._build_chrf()

        all_stats = chrf._extract_corpus_statistics(hyps, [refs])

        for stats in all_stats:
            self.stats += torch.tensor(stats, device=self.device, dtype=torch.int64)

        return self

    @override
    @torch.inference_mode()
    def compute(self) -> Tensor:
        chrf = self._build_chrf()

        score_output = chrf._compute_score_from_stats(self.stats.tolist())

        return torch.tensor(score_output.score, device=self.device)

    @override
    @torch.inference_mode()
    def merge_state(self, metrics: Iterable[ChrfMetric]) -> Self:
        for metric in metrics:
            self.stats += metric.stats.to(self.device)

        return self

    def _build_chrf(self) -> CHRF:
        return CHRF(char_order=self.CHAR_ORDER, word_order=self.WORD_ORDER)
