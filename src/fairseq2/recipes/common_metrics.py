# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torcheval.metrics import Throughput

from fairseq2.gang import Gang
from fairseq2.generation import Seq2SeqGeneratorOutput, SequenceGeneratorOutput
from fairseq2.metrics import MetricBag
from fairseq2.metrics.aggregation import Max, Mean, Sum
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch


class SequenceMetricBag(MetricBag):
    """Holds the metrics of a sequence model training or evaluation task."""

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
    def update_nll_loss(self, batch: SequenceBatch, loss: Tensor) -> None:
        """Update the NLL loss metric.

        :param batch:
            The batch processed by the model.
        :param nll_loss:
            The loss of ``batch``.
        """
        num_target_elements = batch.num_target_elements()

        normalized_loss = loss / num_target_elements

        self._nll_loss.update(normalized_loss, weight=num_target_elements)

    @torch.inference_mode()
    def update_batch_metrics(self, batch: SequenceBatch) -> None:
        """Update the batch metrics.

        :param batch:
            The batch processed by the model.
        """
        batch_size = batch.batch_size

        num_elements = batch.num_elements()

        num_target_elements = batch.num_target_elements()

        self._batch_size.update(batch_size * self._gang.size)

        self._elements_per_batch.update(num_elements * self._gang.size)

        self._num_examples.update(batch_size)

        self._num_elements.update(num_elements)

        self._num_target_elements.update(num_target_elements)

        self._total_num_examples.update(batch_size)

        self._total_num_elements.update(num_elements)

        self._total_num_target_elements.update(num_target_elements)


class Seq2SeqMetricBag(MetricBag):
    """Holds the metrics of a sequence-to-sequence model training or evaluation task."""

    _nll_loss: Mean
    _batch_size: Mean
    _elements_per_batch: Mean
    _num_examples: Sum
    _num_elements: Sum
    _num_source_elements: Sum
    _num_target_elements: Sum
    _total_num_examples: Sum
    _total_num_elements: Sum
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

        self.register_metric("_num_elements", Sum(device=d), persistent=False)

        self.register_metric("_num_source_elements", Sum(device=d), persistent=False)
        self.register_metric("_num_target_elements", Sum(device=d), persistent=False)

        self._total_num_examples = Sum(device=d)

        self._total_num_elements = Sum(device=d)

        self._total_num_source_elements = Sum(device=d)
        self._total_num_target_elements = Sum(device=d)

    @torch.inference_mode()
    def update_nll_loss(self, batch: Seq2SeqBatch, loss: Tensor) -> None:
        """Update the NLL loss metric.

        :param batch:
            The batch processed by the model.
        :param nll_loss:
            The loss of ``batch``.
        """
        num_target_elements = batch.num_target_elements()

        normalized_loss = loss / num_target_elements

        self._nll_loss.update(normalized_loss, weight=num_target_elements)

    @torch.inference_mode()
    def update_batch_metrics(self, batch: Seq2SeqBatch) -> None:
        """Update the batch metrics.

        :param batch:
            The batch processed by the model.
        """
        batch_size = batch.batch_size

        num_source_elements = batch.num_source_elements()
        num_target_elements = batch.num_target_elements()

        num_elements = num_source_elements + num_target_elements

        self._batch_size.update(batch_size * self._gang.size)

        self._elements_per_batch.update(num_elements * self._gang.size)

        self._num_examples.update(batch_size)

        self._num_elements.update(num_elements)

        self._num_source_elements.update(num_source_elements)
        self._num_target_elements.update(num_target_elements)

        self._total_num_examples.update(batch_size)

        self._total_num_elements.update(num_elements)

        self._total_num_source_elements.update(num_source_elements)
        self._total_num_target_elements.update(num_target_elements)


class SequenceGenerationMetricBag(MetricBag):
    """Holds the metrics of a sequence generation task."""

    _batch_size: Mean
    _elements_per_batch: Mean
    _num_examples: Sum
    _num_elements: Sum
    _generator_prefill_size: Sum
    _generator_num_elements: Sum
    _generator_elements_per_second: Throughput
    _generator_cache_size: Max
    _generator_cache_capacity: Max

    def __init__(self, gang: Gang) -> None:
        """
        :param gang:
            The gang over which to sync metrics.
        """
        super().__init__(gang)

        d = gang.device

        self._batch_size = Mean(device=d)

        self._elements_per_batch = Mean(device=d)

        self._num_examples = Sum(device=d)

        self._num_elements = Sum(device=d)

        self._generator_prefill_size = Sum(device=d)

        self._generator_num_elements = Sum(device=d)

        self._generator_elements_per_second = Throughput(device=d)

        self._generator_cache_size = Max(device=d)

        self._generator_cache_capacity = Max(device=d)

    @torch.inference_mode()
    def update_batch_metrics(self, output: SequenceGeneratorOutput) -> None:
        """Update the batch metrics.

        :param output:
            The output returned by a :class:`SequenceGenerator`.
        """
        batch_size = len(output.hypotheses)

        prefill_size = output.counters.prefill_size

        num_generated_elements = output.counters.num_generated_elements

        num_elements = prefill_size + num_generated_elements

        self._batch_size.update(batch_size * self._gang.size)

        self._elements_per_batch.update(num_elements * self._gang.size)

        self._num_examples.update(batch_size)

        self._num_elements.update(num_elements)

        self._generator_prefill_size.update(prefill_size)

        self._generator_num_elements.update(num_generated_elements)

        self._generator_elements_per_second.update(
            num_generated_elements, output.counters.generation_time
        )

        self._generator_cache_size.update(output.counters.cache_size)

        self._generator_cache_capacity.update(output.counters.cache_capacity)


class Seq2SeqGenerationMetricBag(MetricBag):
    """Holds the metrics of a sequence-to-sequence generation task."""

    _batch_size: Mean
    _elements_per_batch: Mean
    _num_examples: Sum
    _num_elements: Sum
    _num_source_elements: Sum
    _generator_prefill_size: Sum
    _generator_num_elements: Sum
    _generator_elements_per_second: Throughput
    _generator_cache_size: Max
    _generator_cache_capacity: Max

    def __init__(self, gang: Gang) -> None:
        """
        :param gang:
            The gang over which to sync metrics.
        """
        super().__init__(gang)

        d = gang.device

        self._batch_size = Mean(device=d)

        self._elements_per_batch = Mean(device=d)

        self._num_examples = Sum(device=d)

        self._num_elements = Sum(device=d)

        self._num_source_elements = Sum(device=d)

        self._generator_prefill_size = Sum(device=d)

        self._generator_num_elements = Sum(device=d)

        self._generator_elements_per_second = Throughput(device=d)

        self._generator_cache_size = Max(device=d)

        self._generator_cache_capacity = Max(device=d)

    @torch.inference_mode()
    def update_batch_metrics(
        self, output: Seq2SeqGeneratorOutput, num_source_elements: int
    ) -> None:
        """Update the batch metrics.

        :param output:
            The output returned by a :class:`Seq2SeqGenerator`.
        :param num_source_elements:
            The number of source elements processed by the underlying model.
        """
        batch_size = len(output.hypotheses)

        prefill_size = output.counters.prefill_size

        num_generated_elements = output.counters.num_generated_elements

        num_elements = num_source_elements + prefill_size + num_generated_elements

        self._batch_size.update(batch_size * self._gang.size)

        self._elements_per_batch.update(num_elements * self._gang.size)

        self._num_examples.update(batch_size)

        self._num_elements.update(num_elements)

        self._num_source_elements.update(num_source_elements)

        self._generator_prefill_size.update(prefill_size)

        self._generator_num_elements.update(num_generated_elements)

        self._generator_elements_per_second.update(
            num_generated_elements, output.counters.generation_time
        )

        self._generator_cache_size.update(output.counters.cache_size)

        self._generator_cache_capacity.update(output.counters.cache_capacity)


def compute_throughput(
    metric_values: Dict[str, Any],
    throughput_metric_name: Optional[str],
    elapsed_time: float,
) -> None:
    """Computes the task throughput.

    :param metric_values:
        The metric values computed by a :class:`MetricBag`.
    :param throughput_metric_name:
        The name of the throughput metric (e.g. num_elements).
    :param elapsed_time:
        The time elapsed since the last throughput call.
    """
    if throughput_metric_name is None:
        return

    try:
        num_elements = metric_values[throughput_metric_name]
    except KeyError:
        return

    if not isinstance(num_elements, (int, float, Tensor)):
        return

    if elapsed_time == 0.0:
        metric_values["elements_per_second"] = 0.0
    else:
        metric_values["elements_per_second"] = num_elements / elapsed_time
