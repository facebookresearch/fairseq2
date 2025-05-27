# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast, final

from torch import Tensor

from fairseq2.datasets import SequenceBatch
from fairseq2.models.wav2vec2 import Wav2Vec2Model, Wav2Vec2Output
from fairseq2.recipes import Model

# isort: split

from fairseq2.recipes.wav2vec2._metrics import Wav2Vec2Loss, Wav2Vec2MetricBag


@final
class Wav2Vec2Criterion:
    _model: Model
    _diversity_loss_weight: float
    _features_penalty_weight: float

    def __init__(
        self,
        model: Model,
        diversity_loss_weight: float,
        features_penalty_weight: float,
    ) -> None:
        self._model = model

        self._diversity_loss_weight = diversity_loss_weight

        self._features_penalty_weight = features_penalty_weight

    def __call__(
        self, batch: SequenceBatch, metric_bag: Wav2Vec2MetricBag
    ) -> tuple[Tensor, int]:
        seqs, seqs_layout = batch.as_input()

        w2v2_output = self._model(seqs, seqs_layout)

        w2v2_loss = self._compute_loss(w2v2_output)

        batch_size, seq_len = w2v2_output.logits.shape[:2]

        num_targets = batch_size * seq_len

        metric_bag.update_loss(w2v2_loss, num_targets)

        metric_bag.update_accuracy(w2v2_output.logits)

        metric_bag.update_quantizer_metrics(w2v2_output.quantizer_output)

        metric_bag.update_batch_metrics(batch)

        return w2v2_loss.aggregate, num_targets

    def _compute_loss(self, output: Wav2Vec2Output) -> Wav2Vec2Loss:
        logits = output.logits

        base_module = cast(Wav2Vec2Model, self._model.base_module)

        contrastive_loss = base_module.compute_contrastive_loss(logits)

        diversity_loss = base_module.compute_diversity_loss(
            logits, output.quantizer_output.prob_perplexity
        )

        features_penalty = base_module.compute_features_penalty(
            logits, output.raw_features
        )

        weighted_diversity_loss = self._diversity_loss_weight * diversity_loss

        weighted_features_penalty = self._features_penalty_weight * features_penalty

        aggregate_loss = (
            contrastive_loss + weighted_diversity_loss + weighted_features_penalty
        )

        return Wav2Vec2Loss(
            aggregate_loss, contrastive_loss, diversity_loss, features_penalty
        )
