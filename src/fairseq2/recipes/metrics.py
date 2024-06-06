# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch import Tensor
from torcheval.metrics import Mean, Sum

from fairseq2.gang import Gang
from fairseq2.metrics import MetricBag
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch


class SequenceModelMetricBag(MetricBag):
    """Holds the common metrics of a sequence model training."""

    _nll_loss: Mean
    _batch_size: Mean
    _elements_per_batch: Mean
    _num_examples: Sum
    _num_elements: Sum
    _num_target_elements: Sum
    _total_num_examples: Sum
    _total_num_elements: Sum
    _total_num_target_elements: Sum

    def __init__(self, gang: Gang) -> None:
        """
        :param gang:
            The gang over which to sync metrics.
        """
        super().__init__(gang)

        d = gang.device

        self.register_metric("_nll_loss", Mean(device=d), persistent=False)

        self.register_metric("_batch_size", Mean(device=d), persistent=False)

        self.register_metric("_elements_per_batch", Mean(device=d), persistent=False)

        self.register_metric("_num_examples", Sum(device=d), persistent=False)

        self.register_metric("_num_elements", Sum(device=d), persistent=False)

        self.register_metric("_num_target_elements", Sum(device=d), persistent=False)

        self._total_num_examples = Sum(device=d)

        self._total_num_elements = Sum(device=d)

        self._total_num_target_elements = Sum(device=d)

    @torch.inference_mode()
    def update(self, batch: SequenceBatch, nll_loss: Tensor) -> None:
        """Update the metrics.

        :param batch:
            The batch processed by the model.
        :param nll_loss:
            The loss of ``batch``.
        """
        batch_size = torch.tensor(batch.batch_size)

        num_elements = torch.tensor(batch.num_elements())
        num_target_elements = torch.tensor(batch.num_target_elements())

        normalized_nll_loss = nll_loss.cpu() / num_target_elements

        self._nll_loss.update(normalized_nll_loss, weight=num_target_elements)

        self._batch_size.update(batch_size * self._gang.size)

        self._elements_per_batch.update(num_elements * self._gang.size)

        self._num_examples.update(batch_size)

        self._num_elements.update(num_elements)

        self._num_target_elements.update(num_target_elements)

        self._total_num_examples.update(batch_size)

        self._total_num_elements.update(num_elements)

        self._total_num_target_elements.update(num_target_elements)


class Seq2SeqModelMetricBag(MetricBag):
    """Holds the common metrics of a sequence-to-sequence model training."""

    _nll_loss: Mean
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

        self.register_metric("_nll_loss", Mean(device=d), persistent=False)

        self.register_metric("_batch_size", Mean(device=d), persistent=False)

        self.register_metric("_elements_per_batch", Mean(device=d), persistent=False)

        self.register_metric("_num_examples", Sum(device=d), persistent=False)

        self.register_metric("_num_source_elements", Sum(device=d), persistent=False)
        self.register_metric("_num_target_elements", Sum(device=d), persistent=False)

        self._total_num_examples = Sum(device=d)

        self._total_num_source_elements = Sum(device=d)
        self._total_num_target_elements = Sum(device=d)

    @torch.inference_mode()
    def update(self, batch: Seq2SeqBatch, nll_loss: Tensor) -> None:
        """Update the metrics.

        :param batch:
            The batch processed by the model.
        :param nll_loss:
            The loss of ``batch``.
        """
        batch_size = torch.tensor(batch.batch_size)

        num_source_elements = torch.tensor(batch.num_source_elements)
        num_target_elements = torch.tensor(batch.num_target_elements)

        normalized_nll_loss = nll_loss.cpu() / num_target_elements

        self._nll_loss.update(normalized_nll_loss, weight=num_target_elements)

        self._batch_size.update(batch_size * self._gang.size)

        self._elements_per_batch.update(num_target_elements * self._gang.size)

        self._num_examples.update(batch_size)

        self._num_source_elements.update(num_source_elements)
        self._num_target_elements.update(num_target_elements)

        self._total_num_examples.update(batch_size)

        self._total_num_source_elements.update(num_source_elements)
        self._total_num_target_elements.update(num_target_elements)
