# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

from fairseq2.checkpoint import CheckpointHGExporter, OutOfProcCheckpointHGExporter
from fairseq2.metrics.recorders import MetricDescriptor
from fairseq2.optim.fp16_loss_scaler import Float16LossScaler
from fairseq2.recipe.internal.trainer import (
    _CheckpointHGExporterFactory,
    _Float16LossScalerFactory,
    _GarbageCollectorFactory,
    _MaybeScoreMetricProvider,
    _TrainerFactory,
    _ValidatorFactory,
)
from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyResolver,
    wire_object,
)
from fairseq2.trainer import Trainer
from fairseq2.utils.gc import GarbageCollector
from fairseq2.validator import StandardValidator


def _register_trainer_factory(container: DependencyContainer) -> None:
    # Trainer
    def create_trainer_factory(resolver: DependencyResolver) -> _TrainerFactory:
        def create_trainer(**kwargs: Any) -> Trainer:
            return wire_object(resolver, Trainer, **kwargs)

        return wire_object(resolver, _TrainerFactory, base_factory=create_trainer)

    container.register(_TrainerFactory, create_trainer_factory)

    # Validator
    def create_validator_factory(resolver: DependencyResolver) -> _ValidatorFactory:
        def create_validator(**kwargs: Any) -> StandardValidator:
            return wire_object(resolver, StandardValidator, **kwargs)

        return wire_object(
            resolver, _ValidatorFactory, standard_factory=create_validator
        )

    container.register(_ValidatorFactory, create_validator_factory)

    # Loss Scaler
    def create_fp16_loss_scaler(resolver: DependencyResolver) -> Float16LossScaler:
        scaler_factory = resolver.resolve(_Float16LossScalerFactory)

        return scaler_factory.create()

    container.register(Float16LossScaler, create_fp16_loss_scaler)

    container.register_type(_Float16LossScalerFactory)

    # Hugging Face Exporter
    def create_checkpoint_hg_exporter(
        resolver: DependencyResolver,
    ) -> CheckpointHGExporter:
        exporter_factory = resolver.resolve(_CheckpointHGExporterFactory)

        return exporter_factory.create()

    container.register(CheckpointHGExporter, create_checkpoint_hg_exporter)

    def create_checkpoint_hg_exporter_factory(
        resolver: DependencyResolver,
    ) -> _CheckpointHGExporterFactory:
        def create_exporter() -> CheckpointHGExporter:
            return wire_object(resolver, OutOfProcCheckpointHGExporter)

        return wire_object(
            resolver, _CheckpointHGExporterFactory, default_factory=create_exporter
        )

    container.register(
        _CheckpointHGExporterFactory, create_checkpoint_hg_exporter_factory
    )

    # Score Metric
    def maybe_get_score_metric(resolver: DependencyResolver) -> MetricDescriptor | None:
        metric_provider = resolver.resolve(_MaybeScoreMetricProvider)

        return metric_provider.maybe_get()

    container.register(MetricDescriptor, maybe_get_score_metric)

    container.register_type(_MaybeScoreMetricProvider)

    # Garbage Collector
    def create_gc(resolver: DependencyResolver) -> GarbageCollector:
        gc_factory = resolver.resolve(_GarbageCollectorFactory)

        return gc_factory.create()

    container.register(GarbageCollector, create_gc)

    container.register_type(_GarbageCollectorFactory)
