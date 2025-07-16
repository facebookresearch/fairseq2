# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.logging import log
from fairseq2.metrics.recorders import (
    JsonlMetricRecorderHandler,
    LogMetricRecorderHandler,
    MetricDescriptor,
    MetricRecorderHandler,
    TensorBoardRecorderHandler,
    WandbRecorderHandler,
)


def _register_metric_recorders(context: RuntimeContext) -> None:
    registry = context.get_registry(MetricRecorderHandler)

    metric_descriptors = context.get_registry(MetricDescriptor)

    handler: MetricRecorderHandler

    # JSONL
    file_system = context.file_system

    handler = JsonlMetricRecorderHandler(file_system, metric_descriptors)

    registry.register(handler.name, handler)

    # Log
    handler = LogMetricRecorderHandler(log, metric_descriptors)

    registry.register(handler.name, handler)

    # TensorBoard
    handler = TensorBoardRecorderHandler(metric_descriptors)

    registry.register(handler.name, handler)

    # Weights & Biases
    handler = WandbRecorderHandler(file_system, metric_descriptors)

    registry.register(handler.name, handler)
