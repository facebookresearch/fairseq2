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
def update_nll_loss(
    metric_bag: MetricBag, loss: Tensor, num_targets: int | None = None
) -> None:
    loss = loss.detach()

    n = num_targets or 1

    metric_bag.get(Mean, "nll_loss").update(loss / n, weight=n)


@torch.inference_mode()
def update_seq_batch_metrics(metric_bag: MetricBag, batch: SequenceBatch) -> None:
    metric_bag.get(Sum, "num_examples").update(batch.num_examples)

    metric_bag.get(Sum, "num_elements").update(batch.num_elements)

    metric_bag.get(Sum, "num_target_elements").update(batch.num_target_elements)

    metric_bag.get(Sum, "total_num_examples").update(batch.num_examples)

    metric_bag.get(Sum, "total_num_elements").update(batch.num_elements)

    metric_bag.get(Sum, "total_num_target_elements").update(batch.num_target_elements)

    metric_bag.get(Sum, "padding").update(batch.padding)


@torch.inference_mode()
def update_seq2seq_batch_metrics(metric_bag: MetricBag, batch: Seq2SeqBatch) -> None:
    metric_bag.get(Sum, "num_examples").update(batch.num_examples)

    metric_bag.get(Sum, "num_elements").update(batch.num_elements)

    metric_bag.get(Sum, "num_source_elements").update(batch.num_source_elements)
    metric_bag.get(Sum, "num_target_elements").update(batch.num_target_elements)

    metric_bag.get(Sum, "total_num_examples").update(batch.num_examples)

    metric_bag.get(Sum, "total_num_elements").update(batch.num_elements)

    metric_bag.get(Sum, "total_num_source_elements").update(batch.num_source_elements)
    metric_bag.get(Sum, "total_num_target_elements").update(batch.num_target_elements)

    metric_bag.get(Sum, "padding").update(batch.padding)


@torch.inference_mode()
def update_seq_generator_metrics(
    metric_bag: MetricBag, output: SequenceGeneratorOutput
) -> None:
    num_examples = len(output.hypotheses)

    prefill_size = output.counters.prefill_size

    num_generated_elements = output.counters.num_generated_elements

    num_elements = prefill_size + num_generated_elements

    metric_bag.get(Sum, "num_examples").update(num_examples)

    metric_bag.get(Sum, "num_elements").update(num_elements)

    metric_bag.get(Sum, "total_num_examples").update(num_examples)

    metric_bag.get(Sum, "total_num_elements").update(num_elements)

    metric_bag.get(Sum, "generator_prefill_size").update(prefill_size)

    metric_bag.get(Sum, "generator_num_elements").update(num_generated_elements)

    metric_bag.get(Throughput, "generator_elements_per_second").update(
        num_generated_elements, output.counters.generation_time
    )

    metric_bag.get(Max, "generator_cache_size").update(output.counters.cache_size)

    metric_bag.get(Max, "generator_cache_capacity").update(
        output.counters.cache_capacity
    )


@torch.inference_mode()
def update_seq2seq_generator_metrics(
    metric_bag: MetricBag, output: SequenceGeneratorOutput, num_source_elements: int
) -> None:
    num_examples = len(output.hypotheses)

    prefill_size = output.counters.prefill_size

    num_generated_elements = output.counters.num_generated_elements

    num_elements = num_source_elements + prefill_size + num_generated_elements

    metric_bag.get(Sum, "num_examples").update(num_examples)

    metric_bag.get(Sum, "num_elements").update(num_elements)

    metric_bag.get(Sum, "num_source_elements").update(num_source_elements)

    metric_bag.get(Sum, "total_num_examples").update(num_examples)

    metric_bag.get(Sum, "total_num_elements").update(num_elements)

    metric_bag.get(Sum, "total_num_source_elements").update(num_source_elements)

    metric_bag.get(Sum, "generator_prefill_size").update(prefill_size)

    metric_bag.get(Sum, "generator_num_elements").update(num_generated_elements)

    metric_bag.get(Throughput, "generator_elements_per_second").update(
        num_generated_elements, output.counters.generation_time
    )

    metric_bag.get(Max, "generator_cache_size").update(output.counters.cache_size)

    metric_bag.get(Max, "generator_cache_capacity").update(
        output.counters.cache_capacity
    )


def extend_batch_metric_values(
    values: MutableMapping[str, object], num_batches: int, elapsed_time: float
) -> None:
    def get_value(name: str) -> int | float | Tensor | None:
        try:
            value = values[name]
        except KeyError:
            return None

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
