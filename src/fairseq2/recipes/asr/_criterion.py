# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch import Tensor

from fairseq2.datasets import Seq2SeqBatch
from fairseq2.recipes import Model

# isort: split

from fairseq2.recipes.asr._metrics import AsrMetricBag
from fairseq2.recipes.asr._scorer import AsrScorer


@final
class AsrCriterion:
    _model: Model
    _scorer: AsrScorer | None

    def __init__(self, model: Model, scorer: AsrScorer | None = None) -> None:
        self._model = model

        self._scorer = scorer

    def __call__(
        self, batch: Seq2SeqBatch, metric_bag: AsrMetricBag
    ) -> tuple[Tensor, int]:
        source_seqs, source_seqs_layout = batch.as_source_input()
        target_seqs, target_seqs_layout = batch.as_target_input()

        ctc_loss, logits, logits_layout = self._model(
            source_seqs,
            source_seqs_layout,
            target_seqs,
            target_seqs_layout,
            return_logits=True,
        )

        metric_bag.update_ctc_loss(batch, ctc_loss)

        metric_bag.update_batch_metrics(batch)

        if self._scorer is not None:
            self._scorer(batch, logits, logits_layout, metric_bag)

        return ctc_loss, batch.batch_size
