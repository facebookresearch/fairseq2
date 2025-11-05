# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from torch import Tensor

from fairseq2.datasets import SequenceBatch
from fairseq2.metrics import MetricBag
from fairseq2.models.wav2vec2 import Wav2Vec2Loss, Wav2Vec2Model, Wav2Vec2Output
from fairseq2.recipe import RecipeModel

from .metrics import (
    add_ssl_metrics,
    update_wav2vec2_accuracy,
    update_wav2vec2_batch_metrics,
    update_wav2vec2_loss,
    update_wav2vec2_quantizer_metrics,
)


@dataclass(kw_only=True)
class Wav2Vec2SslLossSection:
    """wav2vec2 loss configuration section."""

    diversity_loss_weight: float = 0.1
    """The weight of the diversity loss."""

    feature_penalty_weight: float = 10.0
    """The weight of the regularization penalty applied to the extracted features."""


@final
class Wav2Vec2SslCriterion:
    """wav2vec2 training criterion."""

    _model: RecipeModel
    _diversity_weight: float
    _features_penalty_weight: float

    def __init__(
        self,
        model: RecipeModel,
        diversity_weight: float,
        features_penalty_weight: float,
    ) -> None:
        if not isinstance(model.base_module, Wav2Vec2Model):
            raise TypeError(
                f"`model.base_module` must be of type `{Wav2Vec2Model}`, but is of type `{type(model.base_module)}` instead."
            )

        self._model = model
        self._diversity_weight = diversity_weight
        self._features_penalty_weight = features_penalty_weight

    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        add_ssl_metrics(metric_bag)

    def __call__(
        self, batch: SequenceBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, int]:
        loss, output = self._forward(batch)

        batch_size, seq_len = output.logits.shape[:2]
        num_targets = batch_size * seq_len

        update_wav2vec2_loss(metric_bag, loss, num_targets)
        update_wav2vec2_accuracy(metric_bag, output)
        update_wav2vec2_quantizer_metrics(metric_bag, output.quantizer_output)
        update_wav2vec2_batch_metrics(metric_bag, batch)

        return loss.aggregate, num_targets

    def _forward(self, batch: SequenceBatch) -> tuple[Wav2Vec2Loss, Wav2Vec2Output]:
        """Forward pass through the model."""
        seqs, seqs_layout = batch.as_input()

        return self._model.module(  # type: ignore[no-any-return]
            seqs,
            seqs_layout,
            diversity_weight=self._diversity_weight,
            features_penalty_weight=self._features_penalty_weight,
        )

    @property
    def model(self) -> RecipeModel:
        return self._model
