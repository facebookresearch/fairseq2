# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch import Tensor

from fairseq2.datasets import Seq2SeqBatch
from fairseq2.metrics import MetricBag
from fairseq2.model import Model
from fairseq2.nn import BatchLayout

# isort: split

from .metrics import update_asr_batch_metrics, update_ctc_loss


@final
class Wav2Vec2AsrCriterion:
    """wav2vec2 ASR training criterion with CTC loss."""

    _model: Model

    def __init__(self, model: Model) -> None:
        self._model = model

    def __call__(
        self, batch: Seq2SeqBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, int]:
        """
        Compute CTC loss for ASR training.

        :param batch: The input batch containing audio and text.
        :param metric_bag: The metric bag to update.

        :returns: The CTC loss and batch size.
        """
        ctc_loss = self._forward(batch)

        update_ctc_loss(metric_bag, ctc_loss, batch.batch_size)  # type: ignore
        update_asr_batch_metrics(metric_bag, batch)

        return ctc_loss, batch.batch_size  # type: ignore

    def _forward(self, batch: Seq2SeqBatch) -> Tensor | tuple[Tensor, BatchLayout]:
        """
        :return: Tensor (default) - CTC Loss
        :return: tuple[Tensor, BatchLayout] (if target_seqs == None) - logits, layout
        """
        source_seqs, source_seqs_layout = batch.as_source_input()  # Audio
        target_seqs, target_seqs_layout = batch.as_target_input()  # Text tokens

        return self._model.module(
            source_seqs, source_seqs_layout, target_seqs, target_seqs_layout
        )

    @property
    def model(self) -> Model:
        return self._model
