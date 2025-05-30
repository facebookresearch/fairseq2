# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch import Tensor
from torch.nn import Module

from fairseq2.datasets import Seq2SeqBatch
from fairseq2.metrics import MetricBag
from fairseq2.recipes.metrics import update_nll_loss, update_seq2seq_batch_metrics


@final
class MTCriterion:
    _module: Module
    _label_smoothing: float

    def __init__(self, module: Module, label_smoothing: float = 0.0) -> None:
        self._module = module

        self._label_smoothing = label_smoothing

    def __call__(
        self, batch: Seq2SeqBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, int]:
        batch, target_batch = batch.as_auto_regressive()

        source_seqs, source_seqs_layout = batch.as_source_input()
        target_seqs, target_seqs_layout = batch.as_target_input()

        nll_loss = self._module(
            source_seqs,
            source_seqs_layout,
            target_seqs,
            target_seqs_layout,
            targets=target_batch.seqs,
            label_smoothing=self._label_smoothing,
        )

        update_nll_loss(metric_bag, nll_loss, batch.num_target_elements)

        update_seq2seq_batch_metrics(metric_bag, batch)

        return nll_loss, batch.num_target_elements
