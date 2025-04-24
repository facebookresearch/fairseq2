# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import final

import torch
from torch import Tensor
from torcheval.metrics import MulticlassAccuracy

from fairseq2.gang import Gang
from fairseq2.metrics import Mean
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2 import (
    GumbelVectorQuantizerOutput,
    VectorQuantizerOutput,
    Wav2Vec2Loss,
    Wav2Vec2Model,
    Wav2Vec2Output,
)
from fairseq2.recipes import Model, RecipeMetricBag


@dataclass(kw_only=True)
class Wav2Vec2LossSection:
    diversity_loss_weight: float = 0.1
    """The weight of the diversity loss."""

    feature_penalty_weight: float = 10.0
    """The weight of the regularization penalty applied to the extracted features."""


@final
class Wav2Vec2Criterion:
    _model: Model
    _diversity_loss_weight: float
    _feature_penalty_weight: float

    def __init__(
        self, model: Model, diversity_loss_weight: float, feature_penalty_weight: float
    ) -> None:
        if not isinstance(model.base_module, Wav2Vec2Model):
            raise TypeError(
                f"`model.base_module` must be of type `{Wav2Vec2Model}`, but is of type `{type(model.base_module)}` instead."
            )

        self._model = model

        self._diversity_loss_weight = diversity_loss_weight
        self._feature_penalty_weight = feature_penalty_weight

    def __call__(
        self, batch: SequenceBatch, metric_bag: Wav2Vec2MetricBag
    ) -> tuple[Tensor, int]:
        model_output: Wav2Vec2Output = self._model.module(batch)

        loss = model_output.compute_loss(
            self._diversity_loss_weight, self._feature_penalty_weight
        )

        batch_size, seq_len = model_output.logits.shape[:2]

        num_targets = batch_size * seq_len

        metric_bag.update_losses(loss, num_targets)

        metric_bag.update_accuracy(model_output)

        metric_bag.update_quantizer_metrics(model_output.quantizer_output)

        metric_bag.update_batch_metrics(batch)

        return loss.total, num_targets

    @property
    def model(self) -> Model:
        return self._model


class Wav2Vec2MetricBag(RecipeMetricBag):
    loss: Mean
    contrastive_loss: Mean
    diversity_loss: Mean
    feature_penalty: Mean
    accuracy: MulticlassAccuracy
    code_perplexity: Mean
    prob_perplexity: Mean
    temperature: Mean

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang)

        device = gang.device

        self.loss = Mean(device=device)

        self.contrastive_loss = Mean(device=device)

        self.diversity_loss = Mean(device=device)

        self.feature_penalty = Mean(device=device)

        self.accuracy = MulticlassAccuracy(device=device)

        self.code_perplexity = Mean(device=device)

        self.prob_perplexity = Mean(device=device)

        self.temperature = Mean(device=device)

    @torch.inference_mode()
    def update_losses(self, loss: Wav2Vec2Loss, num_targets: int) -> None:
        n = num_targets

        d = num_targets * math.log(2)

        self.loss.update(loss.total.detach() / d, weight=n)

        self.contrastive_loss.update(loss.contrastive.detach() / d, weight=n)

        self.diversity_loss.update(loss.diversity.detach() / d, weight=n)

        self.feature_penalty.update(loss.feature_penalty.detach() / d, weight=n)

    @torch.inference_mode()
    def update_accuracy(self, output: Wav2Vec2Output) -> None:
        # (N x S)
        predictions = output.logits.argmax(-1).view(-1)

        # wav2vec2 treats logit at index 0 as the target.
        targets = torch.zeros_like(predictions)

        self.accuracy.update(predictions, targets)

    @torch.inference_mode()
    def update_quantizer_metrics(self, output: VectorQuantizerOutput) -> None:
        if not isinstance(output, GumbelVectorQuantizerOutput):
            return

        self.code_perplexity.update(output.code_perplexity)
        self.prob_perplexity.update(output.prob_perplexity)

        self.temperature.update(output.temperature)

    @torch.inference_mode()
    def update_batch_metrics(self, batch: SequenceBatch) -> None:
        num_examples = batch.batch_size

        num_elements = batch.num_elements()

        self.num_examples.update(num_examples)
        self.num_elements.update(num_elements)

        self.total_num_examples.update(num_examples)
        self.total_num_elements.update(num_elements)
