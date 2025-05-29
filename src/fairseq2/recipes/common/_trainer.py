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
from fairseq2.device import SupportsDeviceTransfer
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.metrics import MetricDescriptor, UnknownMetricDescriptorError
from fairseq2.optim.lr_scheduler import LRScheduler
from fairseq2.recipes import EvalUnit, Trainer, TrainUnit, Validator
from fairseq2.recipes.config import CommonSection, RegimeSection, TrainerSection
from fairseq2.utils.gc import (
    CPythonGarbageCollector,
    GarbageCollector,
    NoopGarbageCollector,
)

# isort: split

from fairseq2.recipes.common._device import create_device_stat_tracker
from fairseq2.recipes.common._metrics import create_metric_recorder
from fairseq2.recipes.common._profilers import create_profiler

BatchT = TypeVar("BatchT", bound=SupportsDeviceTransfer)


def create_trainer(
    context: RuntimeContext,
    trainer_section: TrainerSection,
    regime_section: RegimeSection,
    common_section: CommonSection,
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
    *,
    hyper_params: object = None,
    score_metric: str | None = None,
) -> Trainer:
    score_metric_descriptor = _get_score_metric_descriptor(context, score_metric)

    metric_recorder = create_metric_recorder(
        context, common_section, gangs, output_dir, hyper_params
    )

    profiler = create_profiler(context, common_section, gangs, output_dir)

    garbage_collector = _create_garbage_collector(context, trainer_section)

    device_stat_tracker = create_device_stat_tracker(gangs)

    # TODO: Fix once we support static mixed precision on single device.
    if trainer_section.mixed_precision == "static":
        amp = gangs.root.size == 1 or trainer_section.data_parallelism != "fsdp"
    else:
        amp = trainer_section.mixed_precision == "dynamic"

    if gangs.root.device.type == "cpu":
        log.warning("Based on your environment setup the training will be run on CPU. If this was not intended, check your job options (e.g. pass `--gpus-per-node` on Slurm).")  # fmt: skip

    if valid_units:
        validator = Validator(
            units=valid_units,
            data_readers=valid_data_readers,
            gangs=gangs,
            dtype=trainer_section.dtype,
            amp=amp,
            score_metric_descriptor=score_metric_descriptor,
            checkpoint_manager=checkpoint_manager,
            seed=seed,
            metric_recorder=metric_recorder,
            profiler=profiler,
            device_stat_tracker=device_stat_tracker,
            wall_watch=context.wall_watch,
            progress_reporter=context.progress_reporter,
        )
    else:
        validator = None

    # fmt: off
    return Trainer(
        unit=unit,
        data_reader=data_reader,
        gangs=gangs,
        dtype=trainer_section.dtype,
        amp=amp,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        fp16_loss_scale=trainer_section.fp16_loss_scale,
        no_sync_grad_accumulation=trainer_section.grad_accumulation.no_sync,
        max_grad_norm=trainer_section.max_grad_norm,
        grad_check=trainer_section.grad_check,
        anomaly_detection=trainer_section.anomaly_detection,
        seed=seed,
        max_num_steps=regime_section.num_steps,
        max_num_data_epochs=regime_section.num_data_epochs,
        validator=validator,
        validate_at_start=regime_section.validate_at_start,
        validate_after_n_steps=regime_section.validate_after_n_steps,
        validate_every_n_steps=regime_section.validate_every_n_steps,
        validate_after_n_data_epochs=regime_section.validate_after_n_data_epochs,
        validate_every_n_data_epochs=regime_section.validate_every_n_data_epochs,
        score_metric_descriptor=score_metric_descriptor,
        checkpoint_manager=checkpoint_manager,
        checkpoint_after_n_steps=regime_section.checkpoint_after_n_steps,
        checkpoint_every_n_steps=regime_section.checkpoint_every_n_steps,
        checkpoint_after_n_data_epochs=regime_section.checkpoint_after_n_data_epochs,
        checkpoint_every_n_data_epochs=regime_section.checkpoint_every_n_data_epochs,
        save_model_only=regime_section.save_model_only,
        keep_last_n_checkpoints=regime_section.keep_last_n_checkpoints,
        keep_best_n_checkpoints=regime_section.keep_best_n_checkpoints,
        keep_checkpoint_every_n_steps=regime_section.keep_checkpoint_every_n_steps,
        metric_recorder=metric_recorder,
        publish_metrics_after_n_steps=regime_section.publish_metrics_after_n_steps,
        publish_metrics_every_n_steps=regime_section.publish_metrics_every_n_steps,
        publish_metrics_after_n_data_epochs=regime_section.publish_metrics_after_n_data_epochs,
        publish_metrics_every_n_data_epochs=regime_section.publish_metrics_every_n_data_epochs,
        garbage_collector=garbage_collector,
        profiler=profiler,
        device_stat_tracker=device_stat_tracker,
        wall_watch=context.wall_watch,
        progress_reporter=context.progress_reporter,
    )
    # fmt: on


def _get_score_metric_descriptor(
    context: RuntimeContext, score_metric: str | None
) -> MetricDescriptor | None:
    if score_metric is None:
        return None

    metric_descriptors = context.get_registry(MetricDescriptor)

    try:
        return metric_descriptors.get(score_metric)
    except LookupError:
        raise UnknownMetricDescriptorError(score_metric) from None


def _create_garbage_collector(
    context: RuntimeContext, trainer_section: TrainerSection
) -> GarbageCollector:
    if trainer_section.gc_every_n_steps is None:
        return NoopGarbageCollector()

    return CPythonGarbageCollector(trainer_section.gc_every_n_steps)
