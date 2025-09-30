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
    _MaybeScoreMetricDescriptorProvider,
    _RecipeCheckpointHGExporterFactory,
    _RecipeFloat16LossScalerFactory,
    _RecipeGarbageCollectorFactory,
    _RecipeTrainerFactory,
    _RecipeValidatorFactory,
)
from fairseq2.recipe.trainer import Trainer
from fairseq2.recipe.validator import StandardValidator
from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyResolver,
    wire_object,
)
from fairseq2.utils.gc import GarbageCollector


def _register_trainer_factory(container: DependencyContainer) -> None:
    # Trainer
    def create_trainer_factory(resolver: DependencyResolver) -> _RecipeTrainerFactory:
        def create_trainer(**kwargs: Any) -> Trainer:
            return wire_object(resolver, Trainer, **kwargs)

        return wire_object(
            resolver, _RecipeTrainerFactory, inner_factory=create_trainer
        )

    container.register(_RecipeTrainerFactory, create_trainer_factory)

    # Validator
    def create_validator_factory(
        resolver: DependencyResolver,
    ) -> _RecipeValidatorFactory:
        def create_validator(**kwargs: Any) -> StandardValidator:
            return wire_object(resolver, StandardValidator, **kwargs)

        return wire_object(
            resolver, _RecipeValidatorFactory, standard_factory=create_validator
        )

    container.register(_RecipeValidatorFactory, create_validator_factory)

    # Loss Scaler
    def create_fp16_loss_scaler(resolver: DependencyResolver) -> Float16LossScaler:
        scaler_factory = resolver.resolve(_RecipeFloat16LossScalerFactory)

        return scaler_factory.create()

    container.register(Float16LossScaler, create_fp16_loss_scaler)

    container.register_type(_RecipeFloat16LossScalerFactory)

    # Hugging Face
    def create_checkpoint_hg_exporter(
        resolver: DependencyResolver,
    ) -> CheckpointHGExporter:
        exporter_factory = resolver.resolve(_RecipeCheckpointHGExporterFactory)

        return exporter_factory.create()

    container.register(CheckpointHGExporter, create_checkpoint_hg_exporter)

    container.register_type(_RecipeCheckpointHGExporterFactory)

    container.register_type(OutOfProcCheckpointHGExporter)

    # Score Metric
    def maybe_get_metric_descriptor(
        resolver: DependencyResolver,
    ) -> MetricDescriptor | None:
        metric_provider = resolver.resolve(_MaybeScoreMetricDescriptorProvider)

        return metric_provider.maybe_get()

    container.register(MetricDescriptor, maybe_get_metric_descriptor)

    container.register_type(_MaybeScoreMetricDescriptorProvider)

    # GarbageCollector
    def create_gc(resolver: DependencyResolver) -> GarbageCollector:
        gc_factory = resolver.resolve(_RecipeGarbageCollectorFactory)

        return gc_factory.create()

    container.register(GarbageCollector, create_gc)

    container.register_type(_RecipeGarbageCollectorFactory)
