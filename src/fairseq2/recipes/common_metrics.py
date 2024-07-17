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
from fairseq2.metrics.aggregation import Max, MaxSum, Mean, Sum
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.typing import override


class TaskMetricBag(MetricBag):
    """Holds the metrics of a machine learning task."""

    _train: bool
    _num_batches: MaxSum
    _num_examples: Sum
    _num_elements: Sum
    _total_num_examples: Optional[Sum]
    _total_num_elements: Optional[Sum]

    def __init__(self, gang: Gang, train: bool) -> None:
        """
        :para train:
            If ``True``, indicates that this bag is used in a training task.
        """
        super().__init__(gang)

        d = gang.device

        self._train = train

        self.register_metric("_num_batches", MaxSum(device=d), persistent=False)

        self.register_metric("_num_examples", Sum(device=d), persistent=False)
        self.register_metric("_num_elements", Sum(device=d), persistent=False)

        if train:
            self._total_num_examples = Sum(device=d)
            self._total_num_elements = Sum(device=d)
        else:
            self._total_num_examples = None
            self._total_num_elements = None

    @override
    def process_metric_values(self, values: Dict[str, Any]) -> None:
        super().process_metric_values(values)

        num_batches = values.pop("num_batches")

        num_examples = values["num_examples"]
        num_elements = values["num_elements"]

        values["batch_size"] = num_examples // num_batches

        values["elements_per_batch"] = num_elements // num_batches


class SequenceMetricBag(TaskMetricBag):
    """Holds the metrics of a sequence model training or evaluation task."""

    _nll_loss: Mean
    _num_target_elements: Sum
    _total_num_target_elements: Optional[Sum]

    def __init__(self, gang: Gang, train: bool = True) -> None:
        super().__init__(gang, train=train)

        d = gang.device

        self.register_metric("_nll_loss", Mean(device=d), persistent=False)

        self.register_metric("_num_target_elements", Sum(device=d), persistent=False)

        if train:
            self._total_num_target_elements = Sum(device=d)
        else:
            self._total_num_target_elements = None

    @torch.inference_mode()
    def update_nll_loss(self, batch: SequenceBatch, loss: Tensor) -> None:
        """Update the NLL loss metric.

        :param batch:
            The batch processed by the model.
        :param nll_loss:
            The loss of ``batch``.
        """
        num_target_elements = batch.num_target_elements()

        self._nll_loss.update(loss / num_target_elements, weight=num_target_elements)

    @torch.inference_mode()
    def update_batch_metrics(self, batch: SequenceBatch) -> None:
        """Update the batch metrics.

        :param batch:
            The batch processed by the model.
        """
        num_examples = batch.batch_size
        num_elements = batch.num_elements()

        num_target_elements = batch.num_target_elements()

        self._num_batches.update(1)

        self._num_examples.update(num_examples)
        self._num_elements.update(num_elements)

        self._num_target_elements.update(num_target_elements)

        if self._train:
            assert self._total_num_examples is not None
            assert self._total_num_elements is not None
            assert self._total_num_target_elements is not None

            self._total_num_examples.update(num_examples)
            self._total_num_elements.update(num_elements)

            self._total_num_target_elements.update(num_target_elements)


class Seq2SeqMetricBag(TaskMetricBag):
    """Holds the metrics of a sequence-to-sequence model training or evaluation task."""

    _nll_loss: Mean
    _num_source_elements: Sum
    _num_target_elements: Sum
    _total_num_source_elements: Optional[Sum]
    _total_num_target_elements: Optional[Sum]

    def __init__(self, gang: Gang, train: bool = True) -> None:
        super().__init__(gang, train=train)

        d = gang.device

        self.register_metric("_nll_loss", Mean(device=d), persistent=False)

        self.register_metric("_num_source_elements", Sum(device=d), persistent=False)
        self.register_metric("_num_target_elements", Sum(device=d), persistent=False)

        if train:
            self._total_num_source_elements = Sum(device=d)
            self._total_num_target_elements = Sum(device=d)
        else:
            self._total_num_source_elements = None
            self._total_num_target_elements = None

    @torch.inference_mode()
    def update_nll_loss(self, batch: Seq2SeqBatch, loss: Tensor) -> None:
        """Update the NLL loss metric.

        :param batch:
            The batch processed by the model.
        :param nll_loss:
            The loss of ``batch``.
        """
        num_target_elements = batch.num_target_elements()

        self._nll_loss.update(loss / num_target_elements, weight=num_target_elements)

    @torch.inference_mode()
    def update_batch_metrics(self, batch: Seq2SeqBatch) -> None:
        """Update the batch metrics.

        :param batch:
            The batch processed by the model.
        """
        num_examples = batch.batch_size

        num_source_elements = batch.num_source_elements()
        num_target_elements = batch.num_target_elements()

        num_elements = num_source_elements + num_target_elements

        self._num_batches.update(1)

        self._num_examples.update(num_examples)
        self._num_elements.update(num_elements)

        self._num_source_elements.update(num_source_elements)
        self._num_target_elements.update(num_target_elements)

        if self._train:
            assert self._total_num_examples is not None
            assert self._total_num_elements is not None

            assert self._total_num_source_elements is not None
            assert self._total_num_target_elements is not None

            self._total_num_examples.update(num_examples)
            self._total_num_elements.update(num_elements)

            self._total_num_source_elements.update(num_source_elements)
            self._total_num_target_elements.update(num_target_elements)


class SequenceGenerationMetricBag(TaskMetricBag):
    """Holds the metrics of a sequence generation task."""

    _generator_prefill_size: Sum
    _generator_num_elements: Sum
    _generator_elements_per_second: Throughput
    _generator_cache_size: Max
    _generator_cache_capacity: Max

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang, train=False)

        d = gang.device

        self._generator_prefill_size = Sum(device=d)

        self._generator_num_elements = Sum(device=d)

        self._generator_elements_per_second = Throughput(device=d)

        self._generator_cache_size = Max(device=d)

        self._generator_cache_capacity = Max(device=d)

    @torch.inference_mode()
    def update_batch_metrics(self, output: SequenceGeneratorOutput) -> None:
        """Update the batch metrics.

        :param output:
            The :class:`SequenceGenerator` output.
        """
        num_examples = len(output.hypotheses)

        prefill_size = output.counters.prefill_size

        num_generated_elements = output.counters.num_generated_elements

        num_elements = prefill_size + num_generated_elements

        self._num_batches.update(1)

        self._num_examples.update(num_examples)
        self._num_elements.update(num_elements)

        self._generator_prefill_size.update(prefill_size)

        self._generator_num_elements.update(num_generated_elements)

        self._generator_elements_per_second.update(
            num_generated_elements, output.counters.generation_time
        )

        self._generator_cache_size.update(output.counters.cache_size)

        self._generator_cache_capacity.update(output.counters.cache_capacity)


class Seq2SeqGenerationMetricBag(TaskMetricBag):
    """Holds the metrics of a sequence-to-sequence generation task."""

    _num_source_elements: Sum
    _generator_prefill_size: Sum
    _generator_num_elements: Sum
    _generator_elements_per_second: Throughput
    _generator_cache_size: Max
    _generator_cache_capacity: Max

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang, train=False)

        d = gang.device

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
            The :class:`Seq2SeqGenerator` output.
        :param num_source_elements:
            The number of source elements processed by the model.
        """
        num_examples = len(output.hypotheses)

        prefill_size = output.counters.prefill_size

        num_generated_elements = output.counters.num_generated_elements

        num_elements = num_source_elements + prefill_size + num_generated_elements

        self._num_batches.update(1)

        self._num_examples.update(num_examples)
        self._num_elements.update(num_elements)

        self._num_source_elements.update(num_source_elements)

        self._generator_prefill_size.update(prefill_size)

        self._generator_num_elements.update(num_generated_elements)

        self._generator_elements_per_second.update(
            num_generated_elements, output.counters.generation_time
        )

        self._generator_cache_size.update(output.counters.cache_size)

        self._generator_cache_capacity.update(output.counters.cache_capacity)


def set_throughput_value(metric_values: Dict[str, Any], elapsed_time: float) -> None:
    """Set the throughput value in ``metric_values``."""
    try:
        num_elements = metric_values["num_elements"]
    except KeyError:
        return

    if not isinstance(num_elements, (int, float, Tensor)):
        return

    if elapsed_time == 0.0:
        metric_values["elements_per_second"] = 0.0
    else:
        metric_values["elements_per_second"] = num_elements / elapsed_time
