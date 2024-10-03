# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fairseq2.dependency import DependencyContainer, DependencyResolver
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.metrics import (
    JsonFileMetricRecorder,
    LogMetricRecorder,
    MetricRecorder,
    TensorBoardRecorder,
    WandbRecorder,
)
from fairseq2.recipes.config_manager import (
    ConfigError,
    ConfigManager,
    ConfigNotFoundError,
)


@dataclass(kw_only=True)
class MetricRecordersConfig:
    jsonl: bool = True
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str | None = None


def register_metric_recorders(container: DependencyContainer) -> None:
    container.register_factory(MetricRecorder, _create_log_recorder)
    container.register_factory(MetricRecorder, _create_jsonl_recorder)
    container.register_factory(MetricRecorder, _create_tb_recorder)
    container.register_factory(MetricRecorder, _create_wandb_recorder)


def _create_log_recorder(resolver: DependencyResolver) -> MetricRecorder | None:
    gang = resolver.resolve(Gang)
    if gang.rank != 0:
        return None

    log = get_log_writer("fairseq2.recipes.metrics")

    return LogMetricRecorder(log)


def _create_jsonl_recorder(resolver: DependencyResolver) -> MetricRecorder | None:
    gang = resolver.resolve(Gang)
    if gang.rank != 0:
        return None

    config_manager = resolver.resolve(ConfigManager)

    try:
        config = config_manager.get_config("metric_recorders", MetricRecordersConfig)
    except ConfigNotFoundError:
        config = MetricRecordersConfig()

    if not config.jsonl:
        return None

    output_dir = config_manager.get_config("output_dir", Path).joinpath("metrics")

    return JsonFileMetricRecorder(output_dir)


def _create_tb_recorder(resolver: DependencyResolver) -> MetricRecorder | None:
    gang = resolver.resolve(Gang)
    if gang.rank != 0:
        return None

    config_manager = resolver.resolve(ConfigManager)

    try:
        config = config_manager.get_config("metric_recorders", MetricRecordersConfig)
    except ConfigNotFoundError:
        config = MetricRecordersConfig()

    if not config.tensorboard:
        return None

    output_dir = config_manager.get_config("output_dir", Path).joinpath("tb")

    return TensorBoardRecorder(output_dir)


def _create_wandb_recorder(resolver: DependencyResolver) -> MetricRecorder | None:
    gang = resolver.resolve(Gang)
    if gang.rank != 0:
        return None

    config_manager = resolver.resolve(ConfigManager)

    try:
        config = config_manager.get_config("metric_recorders", MetricRecordersConfig)
    except ConfigNotFoundError:
        config = MetricRecordersConfig()

    if not config.wandb:
        return None

    if config.wandb_project is None:
        raise ConfigError(
            "'metric_recorders.wandb_project' must be specified when W&B logging is enabled."
        )

    output_dir = config_manager.get_config("output_dir", Path).joinpath("wandb")

    return WandbRecorder(config.wandb_project, output_dir)
