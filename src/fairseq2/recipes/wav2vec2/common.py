# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
from torcheval.metrics import Mean, Sum

from fairseq2.gang import Gang
from fairseq2.metrics import MetricBag
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2 import Wav2Vec2Loss


class Wav2Vec2MetricBag(MetricBag):
    """Holds the training metrics of a wav2vec 2.0 model."""

    _loss: Mean
    _batch_size: Mean
    _elements_per_batch: Mean
    _num_examples: Sum
    _num_source_elements: Sum
    _total_num_examples: Sum
    _total_num_source_elements: Sum

    def __init__(self, gang: Gang) -> None:
        """
        :param gang:
            The gang over which to sync metrics.
        """
        super().__init__(gang)

        d = gang.device

        self.register_metric("_loss", Mean(device=d), persistent=False)

        self.register_metric("_batch_size", Mean(device=d), persistent=False)

        self.register_metric("_elements_per_batch", Mean(device=d), persistent=False)

        self.register_metric("_num_examples", Sum(device=d), persistent=False)

        self.register_metric("_num_source_elements", Sum(device=d), persistent=False)

        self._total_num_examples = Sum(device=d)

        self._total_num_source_elements = Sum(device=d)

    @torch.inference_mode()
    def update_loss(self, batch: SequenceBatch, loss: Wav2Vec2Loss) -> None:
        """Update the loss metric.

        :param batch:
            The batch processed by the model.
        :param loss:
            The loss of ``batch``.
        """
        batch_size = torch.tensor(batch.batch_size)

        normalized_loss = loss.total.cpu() / batch_size / math.log(2)

        self._loss.update(normalized_loss, weight=batch_size)

    @torch.inference_mode()
    def update_batch_metrics(self, batch: SequenceBatch) -> None:
        """Update the batch metrics.

        :param seqs:
            The batch of seqs processed by the model.
        """
        batch_size = torch.tensor(batch.batch_size)

        num_source_elements = torch.tensor(batch.num_elements())

        self._batch_size.update(batch_size * self._gang.size)

        self._elements_per_batch.update(num_source_elements * self._gang.size)

        self._num_examples.update(batch_size)

        self._num_source_elements.update(num_source_elements)

        self._total_num_examples.update(batch_size)

        self._total_num_source_elements.update(num_source_elements)
