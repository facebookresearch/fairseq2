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
from fairseq2.metrics.aggregation import Mean, Sum
from fairseq2.models.seq2seq import Seq2SeqBatch


class Wav2Vec2AsrMetricBag(MetricBag):
    """Holds the metrics of a wav2vec 2.0 ASR model training or evaluation task."""

    _ctc_loss: Mean
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

        self.register_metric("_ctc_loss", Mean(device=d), persistent=False)

        self.register_metric("_batch_size", Mean(device=d), persistent=False)

        self.register_metric("_elements_per_batch", Mean(device=d), persistent=False)

        self.register_metric("_num_examples", Sum(device=d), persistent=False)

        self.register_metric("_num_source_elements", Sum(device=d), persistent=False)
        self.register_metric("_num_target_elements", Sum(device=d), persistent=False)

        self._total_num_examples = Sum(device=d)

        self._total_num_source_elements = Sum(device=d)
        self._total_num_target_elements = Sum(device=d)

    @torch.inference_mode()
    def update_ctc_loss(self, batch: Seq2SeqBatch, loss: Tensor) -> None:
        """Update the CTC loss metric.

        :param batch:
            The batch processed by the model.
        :param ctc_loss:
            The loss of ``batch``.
        """
        normalized_loss = loss / batch.batch_size / math.log(2)

        self._ctc_loss.update(normalized_loss, weight=batch.batch_size)

    @torch.inference_mode()
    def update_batch_metrics(self, batch: Seq2SeqBatch) -> None:
        """Update the batch metrics.

        :param batch:
            The batch processed by the model.
        """
        batch_size = batch.batch_size

        num_source_elements = batch.num_source_elements()
        num_target_elements = batch.num_target_elements()

        self._batch_size.update(batch_size * self._gang.size)

        self._elements_per_batch.update(num_source_elements * self._gang.size)

        self._num_examples.update(batch_size)

        self._num_source_elements.update(num_source_elements)
        self._num_target_elements.update(num_target_elements)

        self._total_num_examples.update(batch_size)

        self._total_num_source_elements.update(num_source_elements)
        self._total_num_target_elements.update(num_target_elements)
