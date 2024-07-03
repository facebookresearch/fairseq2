# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
from torch import Tensor

from fairseq2.gang import Gang
from fairseq2.metrics import MetricBag
from fairseq2.metrics.accuracy import AccuracyMetric
from fairseq2.metrics.aggregation import Mean, Sum
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2 import Wav2Vec2Loss
from fairseq2.models.wav2vec2.vector_quantizer import GumbelVectorQuantizerOutput


class Wav2Vec2MetricBag(MetricBag):
    """Holds the metrics of a wav2vec 2.0 model training or evaluation task."""

    _loss: Mean
    _contrastive_loss: Mean
    _diversity_loss: Mean
    _feature_penalty: Mean
    _accuracy: AccuracyMetric
    _code_perplexity: Mean
    _prob_perplexity: Mean
    _temperature: Mean
    _batch_size: Mean
    _elements_per_batch: Mean
    _num_examples: Sum
    _num_source_elements: Sum
    _num_target_elements: Sum
    _total_num_examples: Sum
    _total_num_source_elements: Sum
    _total_num_target_elements: Sum

    def __init__(self, gang: Gang) -> None:
        """
        :param gang:
            The gang over which to sync metrics.
        """
        super().__init__(gang)

        d = gang.device

        self.register_metric("_loss", Mean(device=d), persistent=False)

        self.register_metric("_contrastive_loss", Mean(device=d), persistent=False)

        self.register_metric("_diversity_loss", Mean(device=d), persistent=False)

        self.register_metric("_feature_penalty", Mean(device=d), persistent=False)

        self.register_metric("_accuracy", AccuracyMetric(device=d), persistent=False)

        self.register_metric("_code_perplexity", Mean(device=d), persistent=False)

        self.register_metric("_prob_perplexity", Mean(device=d), persistent=False)

        self.register_metric("_temperature", Mean(device=d), persistent=False)

        self.register_metric("_batch_size", Mean(device=d), persistent=False)

        self.register_metric("_elements_per_batch", Mean(device=d), persistent=False)

        self.register_metric("_num_examples", Sum(device=d), persistent=False)

        self.register_metric("_num_source_elements", Sum(device=d), persistent=False)
        self.register_metric("_num_target_elements", Sum(device=d), persistent=False)

        self._total_num_examples = Sum(device=d)

        self._total_num_source_elements = Sum(device=d)
        self._total_num_target_elements = Sum(device=d)

    @torch.inference_mode()
    def update_losses(self, loss: Wav2Vec2Loss, num_targets: int) -> None:
        """Update the loss metrics.

        :param loss:
            The loss of ``batch``.
        :param num_targets:
            The number of targets used to compute the loss.
        """
        self._loss.update(loss.total / num_targets / math.log(2), weight=num_targets)

        self._contrastive_loss.update(
            loss.contrastive / num_targets / math.log(2), weight=num_targets
        )

        self._diversity_loss.update(
            loss.diversity / num_targets / math.log(2), weight=num_targets
        )

        self._feature_penalty.update(
            loss.feature_penalty / num_targets / math.log(2),
            weight=num_targets,
        )

    @torch.inference_mode()
    def update_quantizer_metrics(
        self, quantizer_output: GumbelVectorQuantizerOutput
    ) -> None:
        """Update the quantizer metrics metrics.

        :param quantizer_output:
            Output of the Gumbel Vector Quantizer.
        """
        self._code_perplexity.update(quantizer_output.code_perplexity)
        self._prob_perplexity.update(quantizer_output.prob_perplexity)
        self._temperature.update(quantizer_output.temperature)

    @torch.inference_mode()
    def update_accuracy(self, logits: Tensor) -> None:
        if logits.numel() == 0:
            correct_preds, total_preds = 0, 0
        else:
            assert logits.ndim > 1, "Logits tensor should be multidimensional."

            max_preds = logits.argmax(-1) == 0
            min_preds = logits.argmin(-1) == 0
            both_preds = max_preds & min_preds
            correct_preds = max_preds.sum() - both_preds.sum()
            total_preds = max_preds.numel()

        self._accuracy.update(correct_preds, total_preds)

    @torch.inference_mode()
    def update_batch_metrics(self, batch: SequenceBatch, num_targets: int) -> None:
        """Update the batch metrics.

        :param batch:
            The batch of seqs processed by the model.
        :num_targets:
            The number of targets used to compute the loss.
        """
        batch_size = batch.batch_size

        num_source_elements = batch.num_elements()

        self._batch_size.update(batch_size * self._gang.size)

        self._elements_per_batch.update(num_source_elements * self._gang.size)

        self._num_examples.update(batch_size)

        self._num_source_elements.update(num_source_elements)
        self._num_target_elements.update(num_targets)

        self._total_num_examples.update(batch_size)

        self._total_num_source_elements.update(num_source_elements)
        self._total_num_target_elements.update(num_targets)
