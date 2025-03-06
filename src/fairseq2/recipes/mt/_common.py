# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from torch import Tensor

from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.seq2seq import Seq2SeqBatch, as_auto_regressive_input
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.recipes import Model, Seq2SeqMetricBag


@dataclass(kw_only=True)
class MTLossSection:
    label_smoothing: float = 0.1
    """The amount of label smoothing to apply while computing the loss."""


@final
class MTCriterion:
    _model: Model
    _label_smoothing: float

    def __init__(self, model: Model, *, label_smoothing: float = 0.0) -> None:
        if not isinstance(model.base_module, EncoderDecoderModel):
            raise TypeError(
                f"`model.base_module` must be of type `{EncoderDecoderModel}`, but is of type `{type(model.base_module)}` instead."
            )

        self._model = model

        self._label_smoothing = label_smoothing

    def __call__(
        self, batch: Seq2SeqBatch, metric_bag: Seq2SeqMetricBag
    ) -> tuple[Tensor, int]:
        input_batch, target_batch = as_auto_regressive_input(batch)

        output = self._forward(input_batch)

        loss = output.compute_loss(
            target_batch.seqs, label_smoothing=self._label_smoothing
        )

        metric_bag.update_nll_loss(input_batch, loss)

        metric_bag.update_batch_metrics(input_batch)

        return loss, batch.num_target_elements()

    def _forward(self, batch: Seq2SeqBatch) -> SequenceModelOutput:
        return self._model.module(batch)  # type: ignore[no-any-return]

    @property
    def model(self) -> Model:
        return self._model
