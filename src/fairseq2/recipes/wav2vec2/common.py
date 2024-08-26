# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
from torch import Tensor
from torcheval.metrics import MulticlassAccuracy

from fairseq2.gang import Gang
from fairseq2.metrics.aggregation import Mean
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2 import Wav2Vec2Loss
from fairseq2.models.wav2vec2.vector_quantizer import (
    GumbelVectorQuantizerOutput,
    VectorQuantizerOutput,
)
from fairseq2.recipes.common_metrics import TaskMetricBag


class Wav2Vec2MetricBag(TaskMetricBag):
    """Holds the metrics of a wav2vec 2.0 model training or evaluation task."""

    _loss: Mean
    _contrastive_loss: Mean
    _diversity_loss: Mean
    _feature_penalty: Mean
    _accuracy: MulticlassAccuracy
    _code_perplexity: Mean
    _prob_perplexity: Mean
    _temperature: Mean

    def __init__(self, gang: Gang, train: bool = True) -> None:
        super().__init__(gang, train=train)

        d = gang.device

        self.register_metric("_loss", Mean(device=d), persistent=False)

        self.register_metric("_contrastive_loss", Mean(device=d), persistent=False)

        self.register_metric("_diversity_loss", Mean(device=d), persistent=False)

        self.register_metric("_feature_penalty", Mean(device=d), persistent=False)

        self.register_metric(
            "_accuracy", MulticlassAccuracy(device=d), persistent=False
        )

        self.register_metric("_code_perplexity", Mean(device=d), persistent=False)

        self.register_metric("_prob_perplexity", Mean(device=d), persistent=False)

        self.register_metric("_temperature", Mean(device=d), persistent=False)

    @torch.inference_mode()
    def update_losses(self, batch: SequenceBatch, loss: Wav2Vec2Loss) -> None:
        """Update the loss metrics."""
        numel = batch.num_elements()

        n = numel * math.log(2)

        self._loss.update(loss.total.detach() / n, weight=numel)

        self._contrastive_loss.update(loss.contrastive.detach() / n, weight=numel)

        self._diversity_loss.update(loss.diversity.detach() / n, weight=numel)

        self._feature_penalty.update(loss.feature_penalty.detach() / n, weight=numel)

    @torch.inference_mode()
    def update_accuracy(self, logits: Tensor) -> None:
        """Update the prediction accuracy."""
        # (N)
        predictions = logits.argmax(-1)

        # wav2vec2 treats logit at index 0 as the target.
        targets = torch.zeros_like(predictions)

        self._accuracy.update(predictions, targets)

    @torch.inference_mode()
    def update_quantizer_metrics(self, output: VectorQuantizerOutput) -> None:
        """Update the quantizer metrics."""
        if not isinstance(output, GumbelVectorQuantizerOutput):
            return

        self._code_perplexity.update(output.code_perplexity)
        self._prob_perplexity.update(output.prob_perplexity)

        self._temperature.update(output.temperature)

    @torch.inference_mode()
    def update_batch_metrics(self, batch: SequenceBatch) -> None:
        """Update the batch metrics."""
        num_examples = batch.batch_size
        num_elements = batch.num_elements()

        self._num_batches.update(1)

        self._num_examples.update(num_examples)
        self._num_elements.update(num_elements)

        if self._train:
            assert self._total_num_examples is not None
            assert self._total_num_elements is not None

            self._total_num_examples.update(num_examples)
            self._total_num_elements.update(num_elements)
