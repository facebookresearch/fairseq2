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
from fairseq2.model import Model
from fairseq2.models.wav2vec2 import Wav2Vec2Model

# isort: split

from .metrics import (
    update_wav2vec2_accuracy,
    update_wav2vec2_batch_metrics,
    update_wav2vec2_loss,
    update_wav2vec2_quantizer_metrics,
)


@dataclass(kw_only=True)
class Wav2Vec2LossSection:
    """wav2vec2 loss configuration section.

    ORIGINAL: fairseq2:e9fbd6/src/fairseq2/recipes/wav2vec2/_common.py:30-37
    Class: Wav2Vec2LossSection
    """

    diversity_loss_weight: float = 0.1
    """The weight of the diversity loss."""

    feature_penalty_weight: float = 10.0
    """The weight of the regularization penalty applied to the extracted features."""


@final
class Wav2Vec2Criterion:
    """wav2vec2 training criterion.

    ORIGINAL: fairseq2:e9fbd6/src/fairseq2/recipes/wav2vec2/_common.py:40-87
    Class: Wav2Vec2Criterion
    """

    _model: Model
    _diversity_weight: float
    _features_penalty_weight: float

    def __init__(
        self, model: Model, diversity_weight: float, features_penalty_weight: float
    ) -> None:
        # ORIGINAL: _common.py lines 47-56
        if not isinstance(model.base_module, Wav2Vec2Model):
            raise TypeError(
                f"`model.base_module` must be of type `{Wav2Vec2Model}`, but is of type `{type(model.base_module)}` instead."
            )

        self._model = model
        self._diversity_weight = diversity_weight
        self._features_penalty_weight = features_penalty_weight

    def __call__(
        self, batch: SequenceBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, int]:
        # ORIGINAL: _common.py lines 58-79
        loss, output = self._forward(batch)

        batch_size, seq_len = output.logits.shape[:2]
        num_targets = batch_size * seq_len

        # Update metrics using the metric functions
        update_wav2vec2_loss(metric_bag, loss, num_targets)
        update_wav2vec2_accuracy(metric_bag, output)
        update_wav2vec2_quantizer_metrics(metric_bag, output.quantizer_output)
        update_wav2vec2_batch_metrics(metric_bag, batch)

        return loss.aggregate, num_targets

    def _forward(self, batch: SequenceBatch):
        """Forward pass through the model.

        ORIGINAL: _common.py lines 81-82
        """
        seqs, seqs_layout = batch.as_input()

        return self._model.module(
            seqs,
            seqs_layout,
            diversity_weight=self._diversity_weight,
            features_penalty_weight=self._features_penalty_weight,
        )

    @property
    def model(self) -> Model:
        """Get the underlying model.

        ORIGINAL: _common.py lines 84-86
        """
        return self._model
