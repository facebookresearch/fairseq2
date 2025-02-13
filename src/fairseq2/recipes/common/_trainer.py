# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from torch.optim import Optimizer

from fairseq2.checkpoint import CheckpointManager
from fairseq2.context import RuntimeContext
from fairseq2.datasets import DataReader
from fairseq2.gang import Gangs
from fairseq2.metrics import MetricDescriptor, UnknownMetricDescriptorError
from fairseq2.optim.lr_scheduler import LRScheduler
from fairseq2.recipes.common._device import create_device_stat_tracker
from fairseq2.recipes.common._metrics import create_metric_recorder
from fairseq2.recipes.common._profilers import create_profiler
from fairseq2.recipes.config import RegimeSection, TrainerSection, get_config_section
from fairseq2.recipes.evaluator import EvalUnit
from fairseq2.recipes.trainer import Trainer, TrainUnit
from fairseq2.registry import Provider
from fairseq2.utils.gc import (
    CPythonGarbageCollector,
    GarbageCollector,
    NoopGarbageCollector,
)

BatchT = TypeVar("BatchT")


def create_trainer(
    context: RuntimeContext,
    recipe_config: object,
    output_dir: Path,
    unit: TrainUnit[BatchT],
    data_reader: DataReader[BatchT],
    valid_units: Sequence[EvalUnit[BatchT]],
    valid_data_readers: Sequence[DataReader[BatchT]],
    gangs: Gangs,
    checkpoint_manager: CheckpointManager,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    seed: int,
) -> Trainer[BatchT]:
    metric_descriptors = context.get_registry(MetricDescriptor)

    score_metric_descriptor = get_score_metric_descriptor(
        recipe_config, metric_descriptors
    )

    metric_recorder = create_metric_recorder(context, recipe_config, output_dir)

    profiler = create_profiler(context, recipe_config, gangs, output_dir)

    trainer_section = get_config_section(recipe_config, "trainer", TrainerSection)

    garbage_collector: GarbageCollector

    if trainer_section.gc_every_n_steps is None:
        garbage_collector = NoopGarbageCollector()
    else:
        garbage_collector = CPythonGarbageCollector(trainer_section.gc_every_n_steps)

    device_stat_tracker = create_device_stat_tracker(gangs)

    # TODO: Fix once we support static mixed precision on single device.
    if trainer_section.mixed_precision == "static":
        amp = gangs.root.size == 1 or trainer_section.data_parallelism != "fsdp"
    else:
        amp = trainer_section.mixed_precision == "dynamic"

    regime_section = get_config_section(recipe_config, "regime", RegimeSection)

    # fmt: off
    return Trainer[BatchT](
        unit=unit,
        data_reader=data_reader,
        gangs=gangs,
        dtype=trainer_section.dtype,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        fp16_loss_scale=trainer_section.fp16_loss_scale,
        max_gradient_norm=trainer_section.max_gradient_norm,
        amp=amp,
        max_num_steps=regime_section.num_steps,
        max_num_data_epochs=regime_section.num_data_epochs,
        score_metric_descriptor=score_metric_descriptor,
        lower_better=regime_section.lower_score_better,
        valid_units=valid_units,
        valid_data_readers=valid_data_readers,
        validate_after_n_steps=regime_section.validate_after_n_steps,
        validate_every_n_steps=regime_section.validate_every_n_steps,
        validate_after_n_data_epochs=regime_section.validate_after_n_data_epochs,
        validate_every_n_data_epochs=regime_section.validate_every_n_data_epochs,
        checkpoint_manager=checkpoint_manager,
        checkpoint_after_n_steps=regime_section.checkpoint_after_n_steps,
        checkpoint_every_n_steps=regime_section.checkpoint_every_n_steps,
        checkpoint_after_n_data_epochs=regime_section.checkpoint_after_n_data_epochs,
        checkpoint_every_n_data_epochs=regime_section.checkpoint_every_n_data_epochs,
        keep_last_n_checkpoints=regime_section.keep_last_n_checkpoints,
        keep_best_n_checkpoints=regime_section.keep_best_n_checkpoints,
        keep_last_n_models=regime_section.keep_last_n_models,
        keep_best_n_models=regime_section.keep_best_n_models,
        metric_recorder=metric_recorder,
        publish_metrics_after_n_steps=regime_section.publish_metrics_after_n_steps,
        publish_metrics_every_n_steps=regime_section.publish_metrics_every_n_steps,
        publish_metrics_after_n_data_epochs=regime_section.publish_metrics_after_n_data_epochs,
        publish_metrics_every_n_data_epochs=regime_section.publish_metrics_every_n_data_epochs,
        garbage_collector=garbage_collector,
        profiler=profiler,
        device_stat_tracker=device_stat_tracker,
        gradient_check=trainer_section.gradient_check,
        anomaly_detection=trainer_section.anomaly_detection,
        wall_watch=context.wall_watch,
        seed=seed,
    )
    # fmt: on


def get_score_metric_descriptor(
    recipe_config: object, metric_descriptors: Provider[MetricDescriptor]
) -> MetricDescriptor | None:
    regime_section = get_config_section(recipe_config, "regime", RegimeSection)

    score_metric = regime_section.score_metric
    if score_metric is None:
        return None

    try:
        return metric_descriptors.get(score_metric)
    except LookupError:
        raise UnknownMetricDescriptorError(score_metric) from None
