# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import MutableMapping
from typing import cast, final

from torch import Tensor
from torch.nn import Module

from fairseq2.datasets import Seq2SeqBatch
from fairseq2.metrics import MetricBag

# isort: split

from fairseq2.recipes.asr._metrics import update_asr_batch_metrics, update_ctc_loss
from fairseq2.recipes.asr._scorer import AsrScorer


@final
class AsrCriterion:
    _module: Module
    _scorer: AsrScorer | None

    def __init__(self, module: Module, scorer: AsrScorer | None = None) -> None:
        self._module = module

        self._scorer = scorer

    def __call__(
        self, batch: Seq2SeqBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, int]:
        source_seqs, source_seqs_layout = batch.as_source_input()
        target_seqs, target_seqs_layout = batch.as_target_input()

        ctc_loss, logits, logits_layout = self._module(
            source_seqs,
            source_seqs_layout,
            target_seqs,
            target_seqs_layout,
            return_logits=True,
        )

        update_ctc_loss(metric_bag, ctc_loss, batch.batch_size)

        update_asr_batch_metrics(metric_bag, batch)

        if self._scorer is not None:
            self._scorer(batch, logits, logits_layout, metric_bag)

        return ctc_loss, batch.batch_size

    def process_metric_values(self, values: MutableMapping[str, object]) -> None:
        value = values.pop("wer")

        uer, wer = cast(tuple[Tensor, Tensor], value)

        if uer >= 1.0 and wer >= 1.0:
            values["uer"] = uer
            values["wer"] = wer
