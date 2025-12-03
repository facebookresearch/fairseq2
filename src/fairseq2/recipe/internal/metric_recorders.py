# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import final

from typing_extensions import override
from wandb import Run as WandbRun
from wandb.errors import UsageError as WandbUsageError

from fairseq2.error import InternalError, OperationalError
from fairseq2.file_system import FileMode, FileSystem
from fairseq2.gang import Gangs
from fairseq2.metrics.recorders import (
    NOOP_METRIC_RECORDER,
    MetricRecorder,
    TensorBoardRecorder,
    WandbRecorder,
)
from fairseq2.recipe.config import CommonSection
from fairseq2.recipe.error import ConfigError
from fairseq2.recipe.internal.config import _RecipeConfigHolder
from fairseq2.utils.env import Environment
from fairseq2.utils.structured import StructureError, ValueConverter


@final
class _MetricRecorderFactory:
    def __init__(
        self, gangs: Gangs, default_factory: Callable[[], MetricRecorder]
    ) -> None:
        self._gangs = gangs
        self._default_factory = default_factory

    def create(self) -> MetricRecorder:
        if self._gangs.root.rank != 0:
            return NOOP_METRIC_RECORDER

        return self._default_factory()


@final
class _MaybeTensorBoardRecorderFactory:
    def __init__(
        self, section: CommonSection, factory: Callable[[], TensorBoardRecorder]
    ) -> None:
        self._section = section
        self._factory = factory

    def maybe_create(self) -> TensorBoardRecorder | None:
        tb_config = self._section.metric_recorders.tensorboard

        if not tb_config.enabled:
            return None

        return self._factory()


@final
class _MaybeWandbRecorderFactory:
    def __init__(
        self, section: CommonSection, factory: Callable[[], WandbRecorder]
    ) -> None:
        self._section = section
        self._factory = factory

    def maybe_create(self) -> WandbRecorder | None:
        wandb_config = self._section.metric_recorders.wandb

        if not wandb_config.enabled:
            return None

        return self._factory()


@final
class _MaybeWandbRunFactory:
    def __init__(
        self,
        section: CommonSection,
        output_dir: Path,
        env: Environment,
        config_holder: _RecipeConfigHolder,
        value_converter: ValueConverter,
        initializer: Callable[..., WandbRun],
        run_id_manager: _WandbRunIdManager,
    ) -> None:
        self._section = section
        self._output_dir = output_dir
        self._env = env
        self._config_holder = config_holder
        self._value_converter = value_converter
        self._initializer = initializer
        self._run_id_manager = run_id_manager

    def maybe_create(self) -> WandbRun | None:
        wandb_config = self._section.metric_recorders.wandb
        if not wandb_config.enabled:
            return None

        try:
            run_id = self._run_id_manager.get_id()
        except OSError as ex:
            raise OperationalError(
                "Failed to initialize Weights & Biases client."
            ) from ex

        try:
            unstructured_config = self._value_converter.unstructure(
                self._config_holder.config
            )
        except StructureError as ex:
            raise InternalError("Recipe configuration cannot be unstructured.") from ex

        if not isinstance(unstructured_config, dict):
            unstructured_config = None

        try:
            return self._initializer(
                entity=wandb_config.entity,
                project=wandb_config.project,
                dir=self._output_dir,
                id=run_id,
                name=wandb_config.run_name,
                config=unstructured_config,
                group=wandb_config.group,
                job_type=wandb_config.job_type,
                resume=wandb_config.resume_mode,
            )
        except WandbUsageError as ex:
            raise ConfigError(
                f"Wrong arguments passed to the Weights & Biases client. {ex}"
            ) from None


class _WandbRunIdManager(ABC):
    @abstractmethod
    def get_id(self) -> str: ...


@final
class _StandardWandbRunIdManager(_WandbRunIdManager):
    def __init__(
        self,
        section: CommonSection,
        env: Environment,
        file_system: FileSystem,
        id_generator: Callable[[], str],
        save_dir: Path,
    ) -> None:
        self._section = section
        self._env = env
        self._file_system = file_system
        self._id_generator = id_generator
        self._save_dir = save_dir

    @override
    def get_id(self) -> str:
        run_id = self._section.metric_recorders.wandb.run_id

        if run_id is None:
            run_id = self._env.maybe_get("WANDB_RUN_ID")
            if run_id is None:
                run_id = self._id_generator()

            return run_id

        if run_id != "persistent":
            return run_id

        run_id_file = self._save_dir.joinpath("wandb_run_id")

        try:
            fp = self._file_system.open_text(run_id_file)
            with fp:
                return fp.read()
        except FileNotFoundError:
            pass

        run_id = self._id_generator()

        fp = self._file_system.open_text(run_id_file, mode=FileMode.WRITE)
        with fp:
            fp.write(run_id)

        return run_id
