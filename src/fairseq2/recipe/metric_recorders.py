# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import final

try:
    import wandb  # type: ignore[import-not-found]
except ImportError:
    _has_wandb = False
else:
    _has_wandb = True

from fairseq2.error import InfraError
from fairseq2.file_system import FileMode, FileSystem
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.metrics.recorders import (
    CompositeMetricRecorder,
    JsonlMetricRecorder,
    LogMetricRecorder,
    MetricDescriptor,
    MetricRecorder,
    NoopMetricRecorder,
    TensorBoardRecorder,
    WandbRecorder,
)
from fairseq2.recipe.config import (
    CommonSection,
    WandbRecorderSection,
    get_config_section,
    get_output_dir,
    get_recipe_config,
)
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.utils.structured import ValueConverter


def _create_metric_recorder(resolver: DependencyResolver) -> MetricRecorder:
    gangs = resolver.resolve(Gangs)

    if gangs.root.rank != 0:
        return NoopMetricRecorder()

    recorders = []

    recorder: MetricRecorder | None

    # Log
    recorder = _create_log_metric_recorder(resolver)

    recorders.append(recorder)

    # JSONL
    recorder = _create_jsonl_metric_recorder(resolver)

    recorders.append(recorder)

    # TensorBoard
    recorder = _maybe_create_tensorboard_recorder(resolver)
    if recorder is not None:
        recorders.append(recorder)

    # Weights & Biases
    recorder = _maybe_create_wandb_recorder(resolver)
    if recorder is not None:
        recorders.append(recorder)

    return CompositeMetricRecorder(recorders)


def _create_log_metric_recorder(resolver: DependencyResolver) -> MetricRecorder:
    metric_descriptors = resolver.get_provider(MetricDescriptor)

    return LogMetricRecorder(log, metric_descriptors)


def _create_jsonl_metric_recorder(resolver: DependencyResolver) -> MetricRecorder:
    file_system = resolver.resolve(FileSystem)

    metric_descriptors = resolver.get_provider(MetricDescriptor)

    output_dir = get_output_dir(resolver)

    metrics_dir = output_dir.joinpath("metrics")

    return JsonlMetricRecorder(metrics_dir, file_system, metric_descriptors)


def _maybe_create_tensorboard_recorder(
    resolver: DependencyResolver,
) -> MetricRecorder | None:
    common_section = get_config_section(resolver, "common", CommonSection)

    section = common_section.metric_recorders.tensorboard

    if not section.enabled:
        return None

    metric_descriptors = resolver.get_provider(MetricDescriptor)

    output_dir = get_output_dir(resolver)

    tb_dir = output_dir.joinpath("tb")

    return TensorBoardRecorder(tb_dir, metric_descriptors)


def _maybe_create_wandb_recorder(resolver: DependencyResolver) -> MetricRecorder | None:
    common_section = get_config_section(resolver, "common", CommonSection)

    section = common_section.metric_recorders.wandb

    if not section.enabled:
        return None

    if not _has_wandb:
        log.warning("wandb not found. Please install it with `pip install wandb`.")  # fmt: skip

        return None

    file_system = resolver.resolve(FileSystem)

    value_converter = resolver.resolve(ValueConverter)

    metric_descriptors = resolver.get_provider(MetricDescriptor)

    output_dir = get_output_dir(resolver)

    recipe_config = get_recipe_config(resolver)

    unstructured_config = value_converter.unstructure(recipe_config)

    if not isinstance(unstructured_config, dict):
        unstructured_config = None

    run_id_manager = _WandbRunIdManager(file_system, wandb.util.generate_id, output_dir)

    run_id = run_id_manager.get_id(common_section.metric_recorders.wandb)

    try:
        run = wandb.init(
            entity=section.entity,
            project=section.project,
            dir=output_dir,
            id=run_id,
            name=section.run_name,
            config=unstructured_config,
            group=section.group,
            job_type=section.job_type,
            resume=section.resume_mode,
        )
    except (RuntimeError, ValueError) as ex:
        raise WandbError(
            "Weights & Biases client cannot be initialized. See the nested exception for details."
        ) from ex

    return WandbRecorder(run, metric_descriptors)


@final
class _WandbRunIdManager:
    _file_system: FileSystem
    _id_generator: Callable[[], str]
    _save_dir: Path

    def __init__(
        self, file_system: FileSystem, id_generator: Callable[[], str], save_dir: Path
    ) -> None:
        self._file_system = file_system
        self._id_generator = id_generator
        self._save_dir = save_dir

    def get_id(self, section: WandbRecorderSection) -> str:
        run_id = section.run_id

        if run_id is None:
            return self._id_generator()

        if run_id != "persistent":
            return run_id

        run_id_file = self._save_dir.joinpath("wandb_run_id")

        try:
            fp = self._file_system.open_text(run_id_file)

            with fp:
                return fp.read()
        except FileNotFoundError:
            pass
        except OSError as ex:
            raise InfraError(
                f"A system error has occurred while reading the Weights & Biases run ID from the '{run_id_file}' file. See the nested exception for details."
            ) from ex

        run_id = self._id_generator()

        try:
            fp = self._file_system.open_text(run_id_file, mode=FileMode.WRITE)

            with fp:
                fp.write(run_id)
        except OSError as ex:
            raise InfraError(
                f"A system error has occurred while writing the Weights & Biases run ID to the '{run_id_file}' file. See the nested exception for details."
            ) from ex

        return run_id


class WandbError(Exception):
    pass
