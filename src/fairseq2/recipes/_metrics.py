# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import MutableMapping

import torch
from torch import Tensor
from torcheval.metrics import Throughput

from fairseq2.gang import Gang
from fairseq2.generation import Seq2SeqGeneratorOutput, SequenceGeneratorOutput
from fairseq2.metrics import Max, Mean, MetricBag, Sum
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch


class BaseMetricBag(MetricBag):
    """Holds the base metrics of a machine learning task."""

    _train: bool

    num_examples: Sum
    num_elements: Sum
    total_num_examples: Sum | None
    total_num_elements: Sum | None

    def __init__(self, gang: Gang, train: bool) -> None:
        """
        :param train:
            If ``True``, indicates that this bag is used in a training task.
        """
        super().__init__(gang)

        d = gang.device

        self._train = train

        self.register_metric("num_examples", Sum(device=d), persistent=False)
        self.register_metric("num_elements", Sum(device=d), persistent=False)

        if train:
            self.total_num_examples = Sum(device=d)
            self.total_num_elements = Sum(device=d)
        else:
            self.total_num_examples = None
            self.total_num_elements = None

    @property
    def train(self) -> bool:
        return self._train


class SequenceMetricBag(BaseMetricBag):
    """Holds the metrics of a sequence model training or evaluation task."""

    nll_loss: Mean
    num_target_elements: Sum
    total_num_target_elements: Sum | None

    def __init__(self, gang: Gang, train: bool = True) -> None:
        super().__init__(gang, train=train)

        d = gang.device

        self.register_metric("nll_loss", Mean(device=d), persistent=False)

        self.register_metric("num_target_elements", Sum(device=d), persistent=False)

        if train:
            self.total_num_target_elements = Sum(device=d)
        else:
            self.total_num_target_elements = None

    @torch.inference_mode()
    def update_nll_loss(
        self, batch: SequenceBatch, loss: Tensor, normalize: bool = True
    ) -> None:
        """Update the NLL loss metric."""
        loss = loss.detach()

        if normalize:
            n = batch.num_target_elements()
        else:
            n = 1

        self.nll_loss.update(loss / n, weight=n)

    @torch.inference_mode()
    def update_batch_metrics(self, batch: SequenceBatch) -> None:
        """Update the batch metrics."""
        num_examples = batch.batch_size

        num_target_elements = batch.num_target_elements()

        num_elements = batch.num_elements()

        self.num_examples.update(num_examples)
        self.num_elements.update(num_elements)

        self.num_target_elements.update(num_target_elements)

        if self._train:
            assert self.total_num_examples is not None
            assert self.total_num_elements is not None

            assert self.total_num_target_elements is not None

            self.total_num_examples.update(num_examples)
            self.total_num_elements.update(num_elements)

            self.total_num_target_elements.update(num_target_elements)


class Seq2SeqMetricBag(BaseMetricBag):
    """Holds the metrics of a sequence-to-sequence model training or evaluation task."""

    nll_loss: Mean
    num_source_elements: Sum
    num_target_elements: Sum
    total_num_source_elements: Sum | None
    total_num_target_elements: Sum | None

    def __init__(self, gang: Gang, train: bool = True) -> None:
        super().__init__(gang, train=train)

        d = gang.device

        self.register_metric("nll_loss", Mean(device=d), persistent=False)

        self.register_metric("num_source_elements", Sum(device=d), persistent=False)
        self.register_metric("num_target_elements", Sum(device=d), persistent=False)

        if train:
            self.total_num_source_elements = Sum(device=d)
            self.total_num_target_elements = Sum(device=d)
        else:
            self.total_num_source_elements = None
            self.total_num_target_elements = None

    @torch.inference_mode()
    def update_nll_loss(
        self, batch: Seq2SeqBatch, loss: Tensor, normalize: bool = True
    ) -> None:
        """Update the NLL loss metric."""
        loss = loss.detach()

        if normalize:
            n = batch.num_target_elements()
        else:
            n = 1

        self.nll_loss.update(loss / n, weight=n)

    @torch.inference_mode()
    def update_batch_metrics(self, batch: Seq2SeqBatch) -> None:
        """Update the batch metrics."""
        num_examples = batch.batch_size

        num_source_elements = batch.num_source_elements()
        num_target_elements = batch.num_target_elements()

        num_elements = num_source_elements + num_target_elements

        self.num_examples.update(num_examples)
        self.num_elements.update(num_elements)

        self.num_source_elements.update(num_source_elements)
        self.num_target_elements.update(num_target_elements)

        if self._train:
            assert self.total_num_examples is not None
            assert self.total_num_elements is not None

            assert self.total_num_source_elements is not None
            assert self.total_num_target_elements is not None

            self.total_num_examples.update(num_examples)
            self.total_num_elements.update(num_elements)

            self.total_num_source_elements.update(num_source_elements)
            self.total_num_target_elements.update(num_target_elements)


class SequenceGenerationMetricBag(BaseMetricBag):
    """Holds the metrics of a sequence generation task."""

    generator_prefill_size: Sum
    generator_num_elements: Sum
    generator_elements_per_second: Throughput
    generator_cache_size: Max
    generator_cache_capacity: Max

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang, train=False)

        d = gang.device

        self.generator_prefill_size = Sum(device=d)

        self.generator_num_elements = Sum(device=d)

        self.generator_elements_per_second = Throughput(device=d)

        self.generator_cache_size = Max(device=d)

        self.generator_cache_capacity = Max(device=d)

    @torch.inference_mode()
    def update_batch_metrics(self, output: SequenceGeneratorOutput) -> None:
        """Update the batch metrics."""
        num_examples = len(output.hypotheses)

        prefill_size = output.counters.prefill_size

        num_generated_elements = output.counters.num_generated_elements

        num_elements = prefill_size + num_generated_elements

        self.num_examples.update(num_examples)
        self.num_elements.update(num_elements)

        self.generator_prefill_size.update(prefill_size)

        self.generator_num_elements.update(num_generated_elements)

        self.generator_elements_per_second.update(
            num_generated_elements, output.counters.generation_time
        )

        self.generator_cache_size.update(output.counters.cache_size)

        self.generator_cache_capacity.update(output.counters.cache_capacity)


class Seq2SeqGenerationMetricBag(BaseMetricBag):
    """Holds the metrics of a sequence-to-sequence generation task."""

    num_source_elements: Sum
    generator_prefill_size: Sum
    generator_num_elements: Sum
    generator_elements_per_second: Throughput
    generator_cache_size: Max
    generator_cache_capacity: Max

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang, train=False)

        d = gang.device

        self.num_source_elements = Sum(device=d)

        self.generator_prefill_size = Sum(device=d)

        self.generator_num_elements = Sum(device=d)

        self.generator_elements_per_second = Throughput(device=d)

        self.generator_cache_size = Max(device=d)

        self.generator_cache_capacity = Max(device=d)

    @torch.inference_mode()
    def update_batch_metrics(
        self, output: Seq2SeqGeneratorOutput, num_source_elements: int
    ) -> None:
        """Update the batch metrics."""
        num_examples = len(output.hypotheses)

        prefill_size = output.counters.prefill_size

        num_generated_elements = output.counters.num_generated_elements

        num_elements = num_source_elements + prefill_size + num_generated_elements

        self.num_examples.update(num_examples)
        self.num_elements.update(num_elements)

        self.num_source_elements.update(num_source_elements)

        self.generator_prefill_size.update(prefill_size)

        self.generator_num_elements.update(num_generated_elements)

        self.generator_elements_per_second.update(
            num_generated_elements, output.counters.generation_time
        )

        self.generator_cache_size.update(output.counters.cache_size)

        self.generator_cache_capacity.update(output.counters.cache_capacity)


def extend_batch_metrics(
    metric_values: MutableMapping[str, object], num_batches: int, elapsed_time: float
) -> None:
    def get_value(name: str) -> int | float | Tensor | None:
        try:
            value = metric_values[name]
        except KeyError:
            return None

        if not isinstance(value, (int, float, Tensor)):
            return None

        return value

    num_examples = get_value("num_examples")
    if num_examples is not None:
        if num_batches > 0:
            metric_values["batch_size"] = num_examples // num_batches
        else:
            metric_values["batch_size"] = 0

    num_elements = get_value("num_elements")
    if num_elements is not None:
        if num_batches > 0:
            metric_values["elements_per_batch"] = num_elements // num_batches
        else:
            metric_values["elements_per_batch"] = 0

        if elapsed_time > 0.0:
            metric_values["elements_per_second"] = num_elements / elapsed_time
        else:
            metric_values["elements_per_second"] = 0.0
