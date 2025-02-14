# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.logging import log
from fairseq2.metrics import (
    MetricDescriptor,
    format_as_byte_size,
    format_as_float,
    format_as_int,
    format_as_percentage,
    format_as_seconds,
)
from fairseq2.metrics.recorders import (
    JSONL_METRIC_RECORDER,
    LOG_METRIC_RECORDER,
    TENSORBOARD_RECORDER,
    WANDB_RECORDER,
    JsonlMetricRecorderHandler,
    LogMetricRecorderHandler,
    MetricRecorderHandler,
    TensorBoardRecorderHandler,
    WandbRecorderHandler,
)


def _register_metric_recorders(context: RuntimeContext) -> None:
    registry = context.get_registry(MetricRecorderHandler)

    handler: MetricRecorderHandler

    metric_descriptors = context.get_registry(MetricDescriptor)

    # Log
    handler = LogMetricRecorderHandler(log, metric_descriptors)

    registry.register(LOG_METRIC_RECORDER, handler)

    # JSONL
    handler = JsonlMetricRecorderHandler(context.file_system, metric_descriptors)

    registry.register(JSONL_METRIC_RECORDER, handler)

    # TensorBoard
    handler = TensorBoardRecorderHandler(metric_descriptors)

    registry.register(TENSORBOARD_RECORDER, handler)

    # Weights & Biases
    handler = WandbRecorderHandler(metric_descriptors)

    registry.register(WANDB_RECORDER, handler)


def _register_metric_descriptors(context: RuntimeContext) -> None:
    registry = context.get_registry(MetricDescriptor)

    def register(descriptor: MetricDescriptor) -> None:
        registry.register(descriptor.name, descriptor)

    # fmt: off
    register(MetricDescriptor("loss",             "Loss",                   90, format_as_float))
    register(MetricDescriptor("contrastive_loss", "Contrastive Loss",      100, format_as_float))
    register(MetricDescriptor("ctc_loss",         "CTC Loss",              100, format_as_float))
    register(MetricDescriptor("diversity_loss",   "Diversity Loss",        100, format_as_float))
    register(MetricDescriptor("nll_loss",         "NLL Loss",              100, format_as_float))
    register(MetricDescriptor("feature_penalty",  "Feature Penalty",       110, format_as_float))
    register(MetricDescriptor("accuracy",         "Accuracy",              200, format_as_float))
    register(MetricDescriptor("bleu",             "BLEU",                  200, format_as_float))
    register(MetricDescriptor("chrf",             "chrF++",                200, format_as_float))
    register(MetricDescriptor("uer",              "Unit Error Rate (UER)", 200, format_as_float))
    register(MetricDescriptor("wer",              "Word Error Rate (WER)", 200, format_as_float))
    register(MetricDescriptor("code_perplexity",  "Code Perplexity",       210, format_as_float))
    register(MetricDescriptor("prob_perplexity",  "Prob Perplexity",       210, format_as_float))
    register(MetricDescriptor("temperature",      "Temperature",           220, format_as_float))
    register(MetricDescriptor("gradient_norm",    "Gradient Norm",         300, format_as_float))
    register(MetricDescriptor("data_epoch",       "Data Epoch",            490, format_as_int))
    register(MetricDescriptor("data_read_time",   "Data Read Time",        500, format_as_seconds))
    register(MetricDescriptor("elapsed_time",     "Elapsed Time",          505, format_as_seconds))
    register(MetricDescriptor("wall_time",        "Wall Time",             510, format_as_seconds))
    register(MetricDescriptor("lr",               "Learning Rate",         700, format_as_float))
    register(MetricDescriptor("loss_scale",       "Loss Scale",            710, format_as_float))

    # Batch Metrics
    register(MetricDescriptor("batch_size",                "Batch Size",                      800, format_as_int))
    register(MetricDescriptor("elements_per_batch",        "Elements per Batch",              800, format_as_int))
    register(MetricDescriptor("elements_per_second",       "Elements per Second",             810, format_as_int))
    register(MetricDescriptor("num_examples",              "Number of Examples",              820, format_as_int))
    register(MetricDescriptor("num_elements",              "Number of Elements",              830, format_as_int))
    register(MetricDescriptor("num_source_elements",       "Number of Source Elements",       830, format_as_int))
    register(MetricDescriptor("num_target_elements",       "Number of Target Elements",       830, format_as_int))
    register(MetricDescriptor("total_num_examples",        "Total Number of Examples",        840, format_as_int))
    register(MetricDescriptor("total_num_elements",        "Total Number of Elements",        850, format_as_int))
    register(MetricDescriptor("total_num_source_elements", "Total Number of Source Elements", 850, format_as_int))
    register(MetricDescriptor("total_num_target_elements", "Total Number of Target Elements", 850, format_as_int))

    # Sequence Generator Metrics
    register(MetricDescriptor("generator_prefill_size",        "Generator/Prefill Size",        900, format_as_int))
    register(MetricDescriptor("generator_num_elements",        "Generator/Number of Elements",  901, format_as_int))
    register(MetricDescriptor("generator_elements_per_second", "Generator/Elements per Second", 902, format_as_int))
    register(MetricDescriptor("generator_cache_size",          "Generator/Cache Size",          903, format_as_byte_size))
    register(MetricDescriptor("generator_cache_capacity",      "Generator/Cache Capacity",      904, format_as_byte_size))

    # Preference Optimization
    register(MetricDescriptor("cpo_loss",         "CPO Loss",                             0, format_as_float))
    register(MetricDescriptor("dpo_loss",         "DPO Loss",                             0, format_as_float))
    register(MetricDescriptor("orpo_loss",        "ORPO Loss",                            0, format_as_float))
    register(MetricDescriptor("simpo_loss",       "SimPO Loss",                           0, format_as_float))
    register(MetricDescriptor("chosen_logps",     "Chosen Sequence Log Probabilities",   50, format_as_float))
    register(MetricDescriptor("rejected_logps",   "Rejected Sequence Log Probabilities", 50, format_as_float))
    register(MetricDescriptor("chosen_lengths",   "Chosen Sequence Length",              70, format_as_float))
    register(MetricDescriptor("rejected_lengths", "Rejected Sequence Length",            70, format_as_float))

    # Memory
    register(MetricDescriptor("peak_active_mem",         "Peak Active Device Memory",       920, format_as_byte_size))
    register(MetricDescriptor("peak_active_mem_ratio",   "Peak Active Device Memory (%)",   920, format_as_percentage))
    register(MetricDescriptor("peak_reserved_mem",       "Peak Reserved Device Memory",     925, format_as_byte_size))
    register(MetricDescriptor("peak_reserved_mem_ratio", "Peak Reserved Device Memory (%)", 925, format_as_percentage))
    # fmt, on
