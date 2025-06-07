# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Sequence

from torch.optim import Optimizer

from fairseq2.checkpoint import (
    CheckpointManager,
    HuggingFaceSaver,
    OutOfProcHuggingFaceSaver,
)
from fairseq2.datasets import DataReader
from fairseq2.evaluator import EvalUnit
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.metrics.recorders import MetricDescriptor
from fairseq2.model.context import ModelContext
from fairseq2.optim.lr_schedulers import LRScheduler
from fairseq2.recipe.config import (
    RegimeSection,
    TrainerSection,
    get_config_section,
    get_output_dir,
)
from fairseq2.recipe.device import _create_device_stat_tracker
from fairseq2.recipe.error import (
    HuggingFaceNotSupportedError,
    UnknownMetricDescriptorError,
)
from fairseq2.recipe.metric_recorders import _create_metric_recorder
from fairseq2.recipe.profilers import _create_profiler
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.trainer import BatchT, Trainer, TrainUnit
from fairseq2.utils.gc import (
    CPythonGarbageCollector,
    GarbageCollector,
    NoopGarbageCollector,
)
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.rng import SeedHolder
from fairseq2.utils.stopwatch import Stopwatch
from fairseq2.utils.threading import ThreadPool
from fairseq2.validator import Validator


def _create_trainer(
    resolver: DependencyResolver,
    unit: TrainUnit[BatchT],
    data_reader: DataReader[BatchT],
    valid_units: Sequence[EvalUnit[BatchT]] | None = None,
    valid_data_readers: Sequence[DataReader[BatchT]] | None = None,
) -> Trainer:
    trainer_section = get_config_section(resolver, "trainer", TrainerSection)

    regime_section = get_config_section(resolver, "regime", RegimeSection)

    seed_holder = resolver.resolve(SeedHolder)

    gangs = resolver.resolve(Gangs)

    optimizer = resolver.resolve(Optimizer)

    lr_scheduler = resolver.resolve(LRScheduler)

    score_metric_descriptor = _maybe_get_score_metric_descriptor(resolver)

    checkpoint_manager = resolver.resolve(CheckpointManager)

    hugging_face_saver = _maybe_create_hugging_face_saver(resolver)

    metric_recorder = _create_metric_recorder(resolver)

    garbage_collector = _create_garbage_collector(resolver)

    profiler = _create_profiler(resolver)

    device_stat_tracker = _create_device_stat_tracker(resolver)

    wall_watch = resolver.resolve(Stopwatch)

    progress_reporter = resolver.resolve(ProgressReporter)

    seed = seed_holder.advance()

    # TODO: Fix once we support static mixed precision on single device.
    if trainer_section.mixed_precision == "static":
        amp = gangs.root.size == 1 or trainer_section.data_parallelism != "fsdp"
    else:
        amp = trainer_section.mixed_precision == "dynamic"

    if gangs.root.device.type == "cpu":
        log.warning("Based on your environment setup the training will be run on CPU. If this was not intended, check your job options (e.g. pass `--gpus-per-node` on Slurm).")  # fmt: skip

    if valid_units:
        if valid_data_readers is None:
            raise ValueError(
                "Both `valid_units` and `valid_data_readers` must be specified."
            )

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
            wall_watch=wall_watch,
            progress_reporter=progress_reporter,
        )
    else:
        validator = None

    seed = seed_holder.advance()

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
        hugging_face_saver=hugging_face_saver,
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
        wall_watch=wall_watch,
        progress_reporter=progress_reporter,
    )
    # fmt: on


def _maybe_create_hugging_face_saver(
    resolver: DependencyResolver,
) -> HuggingFaceSaver | None:
    regime_section = get_config_section(resolver, "regime", RegimeSection)

    if not regime_section.save_as_hugging_face:
        return None

    model_context = resolver.resolve(ModelContext)

    thread_pool = resolver.resolve(ThreadPool)

    output_dir = get_output_dir(resolver)

    if not model_context.handler.supports_hugging_face:
        raise HuggingFaceNotSupportedError(model_context.name)

    checkpoint_dir = output_dir.joinpath("checkpoints")

    return OutOfProcHuggingFaceSaver(checkpoint_dir, thread_pool)


def _maybe_get_score_metric_descriptor(
    resolver: DependencyResolver,
) -> MetricDescriptor | None:
    regime_section = get_config_section(resolver, "regime", RegimeSection)

    score_metric = regime_section.score_metric
    if score_metric is None:
        return None

    metric_descriptors = resolver.get_provider(MetricDescriptor)

    try:
        return metric_descriptors.get(score_metric)
    except LookupError:
        raise UnknownMetricDescriptorError(score_metric) from None


def _create_garbage_collector(resolver: DependencyResolver) -> GarbageCollector:
    trainer_section = get_config_section(resolver, "trainer", TrainerSection)

    if trainer_section.gc_every_n_steps is None:
        return NoopGarbageCollector()

    return CPythonGarbageCollector(trainer_section.gc_every_n_steps)
