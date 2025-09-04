# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

from fairseq2.metrics import (
    format_as_byte_size,
    format_as_float,
    format_as_int,
    format_as_percentage,
    format_as_seconds,
)
from fairseq2.metrics.recorders import (
    CompositeMetricRecorder,
    JsonlMetricRecorder,
    LogMetricRecorder,
    MetricDescriptor,
    MetricDescriptorRegistry,
    MetricRecorder,
    TensorBoardRecorder,
    WandbClient,
    WandbRecorder,
)
from fairseq2.recipe.internal.metric_recorders import (
    _generate_wandb_id,
    _init_wandb,
    _MaybeTensorBoardRecorderFactory,
    _MaybeWandbRecorderFactory,
    _RecipeMetricRecorderFactory,
    _RecipeWandbClientFactory,
    _StandardWandbRunIdManager,
    _WandbIdGenerator,
    _WandbInitializer,
    _WandbRunIdManager,
)
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


def _register_metric_recorders(container: DependencyContainer) -> None:
    _register_metric_descriptors(container)

    # Metric Recorder
    def create_metric_recorder(resolver: DependencyResolver) -> MetricRecorder:
        recorder_factory = resolver.resolve(_RecipeMetricRecorderFactory)

        return recorder_factory.create()

    container.register(MetricRecorder, create_metric_recorder, singleton=True)

    container.register_type(_RecipeMetricRecorderFactory)

    container.register_type(CompositeMetricRecorder)

    # Log & JSONL
    container.collection.register_type(MetricRecorder, LogMetricRecorder)
    container.collection.register_type(MetricRecorder, JsonlMetricRecorder)

    # TensorBoard or None
    def maybe_create_tensorboard_recorder(
        resolver: DependencyResolver,
    ) -> TensorBoardRecorder | None:
        recorder_factory = resolver.resolve(_MaybeTensorBoardRecorderFactory)

        return recorder_factory.maybe_create()

    container.collection.register(MetricRecorder, maybe_create_tensorboard_recorder)

    container.register_type(_MaybeTensorBoardRecorderFactory)

    container.register_type(TensorBoardRecorder)

    # Weights & Biases or None
    def maybe_create_wandb_recorder(
        resolver: DependencyResolver,
    ) -> MetricRecorder | None:
        recorder_factory = resolver.resolve(_MaybeWandbRecorderFactory)

        return recorder_factory.maybe_create()

    container.collection.register(MetricRecorder, maybe_create_wandb_recorder)

    container.register_type(_MaybeWandbRecorderFactory)

    container.register_type(WandbRecorder)

    # Weights & Biases Client
    def create_wandb_client(resolver: DependencyResolver) -> WandbClient:
        client_factory = resolver.resolve(_RecipeWandbClientFactory)

        return client_factory.create()

    container.register(WandbClient, create_wandb_client)

    container.register_type(_RecipeWandbClientFactory)

    container.register_instance(_WandbInitializer, _init_wandb)

    # Run-Id Manager
    container.register_type(_WandbRunIdManager, _StandardWandbRunIdManager)

    container.register_instance(_WandbIdGenerator, _generate_wandb_id)


def _register_metric_descriptors(container: DependencyContainer) -> None:
    container.register_type(MetricDescriptorRegistry)

    # Descriptors
    def register(name: str, *args: Any, **kwargs: Any) -> None:
        descriptor = MetricDescriptor(name, *args, **kwargs)

        container.collection.register_instance(MetricDescriptor, descriptor)

    # fmt: off
    register("loss",             "Loss",                   90, format_as_float)
    register("contrastive_loss", "Contrastive Loss",      100, format_as_float)
    register("ctc_loss",         "CTC Loss",              100, format_as_float)
    register("diversity_loss",   "Diversity Loss",        100, format_as_float)
    register("nll_loss",         "NLL Loss",              100, format_as_float)
    register("features_penalty", "Features Penalty",      110, format_as_float)
    register("accuracy",         "Accuracy",              200, format_as_float, higher_better=True)
    register("bleu",             "BLEU",                  200, format_as_float, higher_better=True)
    register("chrf",             "chrF++",                200, format_as_float, higher_better=True)
    register("uer",              "Unit Error Rate (UER)", 200, format_as_float)
    register("wer",              "Word Error Rate (WER)", 200, format_as_float)
    register("code_perplexity",  "Code Perplexity",       210, format_as_float)
    register("prob_perplexity",  "Prob Perplexity",       210, format_as_float)
    register("temperature",      "Temperature",           220, format_as_float)
    register("grad_norm",        "Gradient Norm",         300, format_as_float)
    register("data_epoch",       "Data Epoch",            490, format_as_int)
    register("data_time",        "Data Time",             500, format_as_seconds)
    register("compute_time",     "Compute Time",          501, format_as_seconds)
    register("lapse_time",       "Lapse Time",            502, format_as_seconds)
    register("total_time",       "Total Time",            505, format_as_seconds)
    register("wall_time",        "Wall Time",             510, format_as_seconds)
    register("lr",               "Learning Rate",         700, format_as_float)
    register("loss_scale",       "Loss Scale",            710, format_as_float)

    # Batch Metrics
    register("batch_size",                "Batch Size",                      800, format_as_int)
    register("elements_per_batch",        "Elements per Batch",              800, format_as_int)
    register("elements_per_second",       "Elements per Second",             810, format_as_int)
    register("num_examples",              "Number of Examples",              820, format_as_int)
    register("num_elements",              "Number of Elements",              830, format_as_int)
    register("num_source_elements",       "Number of Source Elements",       830, format_as_int)
    register("num_target_elements",       "Number of Target Elements",       830, format_as_int)
    register("padding",                   "Padding",                         835, format_as_int)
    register("padding_ratio",             "Padding Ratio (%)",               835, format_as_percentage)
    register("total_num_examples",        "Total Number of Examples",        840, format_as_int)
    register("total_num_elements",        "Total Number of Elements",        850, format_as_int)
    register("total_num_source_elements", "Total Number of Source Elements", 850, format_as_int)
    register("total_num_target_elements", "Total Number of Target Elements", 850, format_as_int)

    # Sequence Generator Metrics
    register("generator_prefill_size",        "Generator/Prefill Size",        900, format_as_int)
    register("generator_num_elements",        "Generator/Number of Elements",  901, format_as_int)
    register("generator_elements_per_second", "Generator/Elements per Second", 902, format_as_int)
    register("generator_cache_size",          "Generator/Cache Size",          903, format_as_byte_size)
    register("generator_cache_capacity",      "Generator/Cache Capacity",      904, format_as_byte_size)

    # Memory
    register("peak_active_mem_bytes",   "Peak Active Device Memory",       920, format_as_byte_size)
    register("peak_active_mem_ratio",   "Peak Active Device Memory (%)",   920, format_as_percentage)
    register("peak_reserved_mem_bytes", "Peak Reserved Device Memory",     925, format_as_byte_size)
    register("peak_reserved_mem_ratio", "Peak Reserved Device Memory (%)", 925, format_as_percentage)
    # fmt: on
