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

from fairseq2.datasets import Seq2SeqBatch, SequenceBatch
from fairseq2.generation import SequenceGeneratorOutput
from fairseq2.metrics.aggregation import Max, Mean, Sum
from fairseq2.metrics.bag import MetricBag


@torch.inference_mode()
def add_nll_loss_metric(metric_bag: MetricBag) -> None:
    metric_bag.add("nll_loss", Mean())


@torch.inference_mode()
def update_nll_loss_metric(
    metric_bag: MetricBag, loss: Tensor, num_targets: int | None = None
) -> None:
    loss = loss.detach()

    n = num_targets or 1

    metric_bag.get("nll_loss", Mean).update(loss / n, weight=n)


@torch.inference_mode()
def add_seq_batch_metrics(metric_bag: MetricBag) -> None:
    metric_bag.add("num_examples", Sum())
    metric_bag.add("num_elements", Sum())
    metric_bag.add("num_target_elements", Sum())
    metric_bag.add("total_num_examples", Sum())
    metric_bag.add("total_num_elements", Sum())
    metric_bag.add("total_num_target_elements", Sum())
    metric_bag.add("padding", Sum())


@torch.inference_mode()
def update_seq_batch_metrics(metric_bag: MetricBag, batch: SequenceBatch) -> None:
    metric_bag.get("num_examples", Sum).update(batch.num_examples)
    metric_bag.get("num_elements", Sum).update(batch.num_elements)
    metric_bag.get("num_target_elements", Sum).update(batch.num_target_elements)
    metric_bag.get("total_num_examples", Sum).update(batch.num_examples)
    metric_bag.get("total_num_elements", Sum).update(batch.num_elements)
    metric_bag.get("total_num_target_elements", Sum).update(batch.num_target_elements)
    metric_bag.get("padding", Sum).update(batch.padding)


@torch.inference_mode()
def add_seq2seq_batch_metrics(metric_bag: MetricBag) -> None:
    metric_bag.add("num_examples", Sum())
    metric_bag.add("num_elements", Sum())
    metric_bag.add("num_source_elements", Sum())
    metric_bag.add("num_target_elements", Sum())
    metric_bag.add("total_num_examples", Sum())
    metric_bag.add("total_num_elements", Sum())
    metric_bag.add("total_num_source_elements", Sum())
    metric_bag.add("total_num_target_elements", Sum())
    metric_bag.add("padding", Sum())


@torch.inference_mode()
def update_seq2seq_batch_metrics(metric_bag: MetricBag, batch: Seq2SeqBatch) -> None:
    metric_bag.get("num_examples", Sum).update(batch.num_examples)
    metric_bag.get("num_elements", Sum).update(batch.num_elements)
    metric_bag.get("num_source_elements", Sum).update(batch.num_source_elements)
    metric_bag.get("num_target_elements", Sum).update(batch.num_target_elements)
    metric_bag.get("total_num_examples", Sum).update(batch.num_examples)
    metric_bag.get("total_num_elements", Sum).update(batch.num_elements)
    metric_bag.get("total_num_source_elements", Sum).update(batch.num_source_elements)
    metric_bag.get("total_num_target_elements", Sum).update(batch.num_target_elements)
    metric_bag.get("padding", Sum).update(batch.padding)


@torch.inference_mode()
def add_seq_generator_metrics(metric_bag: MetricBag) -> None:
    metric_bag.add("num_examples", Sum())
    metric_bag.add("num_elements", Sum())
    metric_bag.add("total_num_examples", Sum())
    metric_bag.add("total_num_elements", Sum())
    metric_bag.add("generator_prefill_size", Sum())
    metric_bag.add("generator_num_elements", Sum())
    metric_bag.add("generator_elements_per_second", Throughput())
    metric_bag.add("generator_cache_size", Max())
    metric_bag.add("generator_cache_capacity", Max())


@torch.inference_mode()
def update_seq_generator_metrics(
    metric_bag: MetricBag, output: SequenceGeneratorOutput
) -> None:
    num_examples = len(output.hypotheses)

    prefill_size = output.counters.prefill_size

    num_generated_elements = output.counters.num_generated_elements

    num_elements = prefill_size + num_generated_elements

    metric_bag.get("num_examples", Sum).update(num_examples)
    metric_bag.get("num_elements", Sum).update(num_elements)
    metric_bag.get("total_num_examples", Sum).update(num_examples)
    metric_bag.get("total_num_elements", Sum).update(num_elements)
    metric_bag.get("generator_prefill_size", Sum).update(prefill_size)
    metric_bag.get("generator_num_elements", Sum).update(num_generated_elements)
    metric_bag.get("generator_elements_per_second", Throughput).update(
        num_generated_elements, output.counters.generation_time
    )
    metric_bag.get("generator_cache_size", Max).update(output.counters.cache_size)
    metric_bag.get("generator_cache_capacity", Max).update(
        output.counters.cache_capacity
    )


@torch.inference_mode()
def add_seq2seq_generator_metrics(metric_bag: MetricBag) -> None:
    metric_bag.add("num_examples", Sum())
    metric_bag.add("num_elements", Sum())
    metric_bag.add("num_source_elements", Sum())
    metric_bag.add("total_num_examples", Sum())
    metric_bag.add("total_num_elements", Sum())
    metric_bag.add("total_num_source_elements", Sum())
    metric_bag.add("generator_prefill_size", Sum())
    metric_bag.add("generator_num_elements", Sum())
    metric_bag.add("generator_elements_per_second", Throughput())
    metric_bag.add("generator_cache_size", Max())
    metric_bag.add("generator_cache_capacity", Max())


@torch.inference_mode()
def update_seq2seq_generator_metrics(
    metric_bag: MetricBag, output: SequenceGeneratorOutput, num_source_elements: int
) -> None:
    num_examples = len(output.hypotheses)

    prefill_size = output.counters.prefill_size

    num_generated_elements = output.counters.num_generated_elements

    num_elements = num_source_elements + prefill_size + num_generated_elements

    metric_bag.get("num_examples", Sum).update(num_examples)
    metric_bag.get("num_elements", Sum).update(num_elements)
    metric_bag.get("num_source_elements", Sum).update(num_source_elements)
    metric_bag.get("total_num_examples", Sum).update(num_examples)
    metric_bag.get("total_num_elements", Sum).update(num_elements)
    metric_bag.get("total_num_source_elements", Sum).update(num_source_elements)
    metric_bag.get("generator_prefill_size", Sum).update(prefill_size)
    metric_bag.get("generator_num_elements", Sum).update(num_generated_elements)
    metric_bag.get("generator_elements_per_second", Throughput).update(
        num_generated_elements, output.counters.generation_time
    )
    metric_bag.get("generator_cache_size", Max).update(output.counters.cache_size)
    metric_bag.get("generator_cache_capacity", Max).update(
        output.counters.cache_capacity
    )


def extend_batch_metric_values(
    values: MutableMapping[str, object], num_batches: int, elapsed_time: float
) -> None:
    def get_value(name: str) -> int | float | Tensor | None:
        value = values.get(name)
        if not isinstance(value, (int, float, Tensor)):
            return None

        return value

    num_examples = get_value("num_examples")
    if num_examples is not None:
        if num_batches > 0:
            values["batch_size"] = num_examples // num_batches
        else:
            values["batch_size"] = 0

    num_elements = get_value("num_elements")
    if num_elements is not None:
        if num_batches > 0:
            values["elements_per_batch"] = num_elements // num_batches
        else:
            values["elements_per_batch"] = 0

        if elapsed_time > 0.0:
            values["elements_per_second"] = num_elements / elapsed_time
        else:
            values["elements_per_second"] = 0.0

        if num_elements > 0:
            padding = get_value("padding")
            if padding is not None:
                values["padding_ratio"] = padding / (num_elements + padding)
