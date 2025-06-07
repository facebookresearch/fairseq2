# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

try:
    import wandb  # type: ignore[import-not-found]
except ImportError:
    _has_wandb = False
else:
    _has_wandb = True

from fairseq2.dependency import DependencyResolver
from fairseq2.error import SetupError
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
from fairseq2.recipe.component import resolve_component
from fairseq2.recipe.config import (
    CommonSection,
    JsonlMetricRecorderConfig,
    LogMetricRecorderConfig,
    TensorBoardRecorderConfig,
    WandbRecorderConfig,
    get_config,
    get_config_section,
    get_output_dir,
)
from fairseq2.utils.structured import StructureError, unstructure


def create_metric_recorder(resolver: DependencyResolver) -> MetricRecorder:
    common_section = get_config_section(resolver, "common", CommonSection)

    gangs = resolver.resolve(Gangs)

    recorders = []

    for name, config in common_section.metric_recorders.items():
        try:
            recorder = resolve_component(resolver, MetricRecorder, name, config)
        except StructureError as ex:
            raise StructureError(
                f"The '{name}' metric recorder configuration cannot be parsed. See the nested exception for details."
            ) from ex

        if gangs.root.rank != 0:
            continue

        recorders.append(recorder)

    return CompositeMetricRecorder(recorders)


def create_jsonl_metric_recorder(
    resolver: DependencyResolver, config: JsonlMetricRecorderConfig
) -> MetricRecorder:
    if not config.enabled:
        return NoopMetricRecorder()

    file_system = resolver.resolve(FileSystem)

    metric_descriptors = resolver.resolve_provider(MetricDescriptor)

    output_dir = get_output_dir(resolver)

    metrics_dir = output_dir.joinpath("metrics")

    return JsonlMetricRecorder(metrics_dir, file_system, metric_descriptors)


def create_log_metric_recorder(
    resolver: DependencyResolver, config: LogMetricRecorderConfig
) -> MetricRecorder:
    if not config.enabled:
        return NoopMetricRecorder()

    metric_descriptors = resolver.resolve_provider(MetricDescriptor)

    return LogMetricRecorder(log, metric_descriptors)


def create_tensorboard_recorder(
    resolver: DependencyResolver, config: TensorBoardRecorderConfig
) -> MetricRecorder:
    if not config.enabled:
        return NoopMetricRecorder()

    metric_descriptors = resolver.resolve_provider(MetricDescriptor)

    output_dir = get_output_dir(resolver)

    tb_dir = output_dir.joinpath("tb")

    return TensorBoardRecorder(tb_dir, metric_descriptors)


def create_wandb_recorder(
    resolver: DependencyResolver, config: WandbRecorderConfig
) -> MetricRecorder:
    if not config.enabled:
        return NoopMetricRecorder()

    if not _has_wandb:
        log.warning("wandb not found. Please install it with `pip install wandb`.")  # fmt: skip

        return NoopMetricRecorder()

    metric_descriptors = resolver.resolve_provider(MetricDescriptor)

    recipe_config = get_config(resolver)

    output_dir = get_output_dir(resolver)

    try:
        recipe_config = unstructure(recipe_config)
    except StructureError as ex:
        raise SetupError(
            "The recipe configuration cannot be unstructured. See the nested exception for details."
        ) from ex

    if not isinstance(recipe_config, dict):
        raise TypeError(
            f"The unstructured form of the recipe configuration is expected to be of type `{dict}`, but is of type `{type(recipe_config)}` instead."
        )

    run_id = _get_wandb_run_id(resolver, config)

    try:
        run = wandb.init(
            entity=config.entity,
            project=config.project,
            dir=output_dir,
            id=run_id,
            name=config.run_name,
            config=recipe_config,
            group=config.group,
            job_type=config.job_type,
            resume=config.resume_mode,
        )
    except (RuntimeError, ValueError) as ex:
        raise SetupError(
            "Weights & Biases client cannot be initialized. See the nested exception for details."
        ) from ex

    return WandbRecorder(run, metric_descriptors)


def _get_wandb_run_id(resolver: DependencyResolver, config: WandbRecorderConfig) -> str:
    file_system = resolver.resolve(FileSystem)

    output_dir = get_output_dir(resolver)

    run_id = config.run_id

    if run_id is None:
        return wandb.util.generate_id()

    if run_id != "auto":
        return run_id

    run_id_file = output_dir.joinpath("wandb_run_id")

    try:
        fp = file_system.open_text(run_id_file)

        with fp:
            return fp.read()
    except FileNotFoundError:
        pass
    except OSError as ex:
        raise SetupError(
            "The Weights & Biases run ID cannot be loaded. See the nested exception for details."
        ) from ex

    run_id = wandb.util.generate_id()

    try:
        fp = file_system.open_text(run_id_file, mode=FileMode.WRITE)

        with fp:
            fp.write(run_id)
    except OSError as ex:
        raise SetupError(
            "The Weights & Biases run ID cannot be saved. See the nested exception for details."
        ) from ex

    return run_id
