# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol, final, runtime_checkable

from typing_extensions import override

try:
    import wandb  # type: ignore[import-not-found]
except ImportError:
    _has_wandb = False
else:
    _has_wandb = True

from fairseq2.error import raise_operational_system_error
from fairseq2.file_system import FileMode, FileSystem
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.metrics.recorders import (
    NOOP_METRIC_RECORDER,
    CompositeMetricRecorder,
    MetricRecorder,
    TensorBoardRecorder,
    WandbClient,
    WandbRecorder,
)
from fairseq2.recipe.config import CommonSection, RecipeConfig
from fairseq2.recipe.error import WandbInitializationError
from fairseq2.utils.structured import ValueConverter


@final
class _RecipeMetricRecorderFactory:
    def __init__(
        self, gangs: Gangs, default_factory: Callable[[], CompositeMetricRecorder]
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

        spec = find_spec("torch.utils.tensorboard")
        if spec is None:
            log.warning("tensorboard is not found. Use `pip install tensorboard`.")

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

        if not _has_wandb:
            log.warning("wandb is not found. Use `pip install wandb`.")

            return None

        return self._factory()


@final
class _RecipeWandbClientFactory:
    def __init__(
        self,
        section: CommonSection,
        output_dir: Path,
        config: RecipeConfig,
        value_converter: ValueConverter,
        initializer: _WandbInitializer,
        run_id_manager: _WandbRunIdManager,
    ) -> None:
        self._section = section
        self._output_dir = output_dir
        self._config = config
        self._value_converter = value_converter
        self._initializer = initializer
        self._run_id_manager = run_id_manager

    def create(self) -> WandbClient:
        untyped_config = self._config.as_(object)

        unstructured_config = self._value_converter.unstructure(untyped_config)

        if not isinstance(unstructured_config, dict):
            unstructured_config = None

        id_ = self._run_id_manager.get_id()

        wandb_config = self._section.metric_recorders.wandb

        try:
            run = self._initializer(
                entity=wandb_config.entity,
                project=wandb_config.project,
                dir=self._output_dir,
                id=id_,
                name=wandb_config.run_name,
                config=unstructured_config,
                group=wandb_config.group,
                job_type=wandb_config.job_type,
                resume=wandb_config.resume_mode,
            )
        except (RuntimeError, ValueError) as ex:
            raise WandbInitializationError() from ex

        return WandbClient(run)


@runtime_checkable
class _WandbInitializer(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def _init_wandb(*args: Any, **kwargs: Any) -> Any:
    return wandb.init(*args, **kwargs)


class _WandbRunIdManager(ABC):
    @abstractmethod
    def get_id(self) -> str: ...


@final
class _StandardWandbRunIdManager(_WandbRunIdManager):
    def __init__(
        self,
        section: CommonSection,
        file_system: FileSystem,
        id_generator: _WandbIdGenerator,
        save_dir: Path,
    ) -> None:
        self._section = section
        self._file_system = file_system
        self._id_generator = id_generator
        self._save_dir = save_dir

    @override
    def get_id(self) -> str:
        run_id = self._section.metric_recorders.wandb.run_id

        if run_id is None:
            return self._id_generator()

        if run_id != "persistent":
            return run_id

        run_id_file = self._save_dir.joinpath("wandb_run_id")

        fp = None

        try:
            fp = self._file_system.open_text(run_id_file)

            return fp.read()
        except FileNotFoundError:
            pass
        except OSError as ex:
            raise_operational_system_error(ex)
        finally:
            if fp is not None:
                fp.close()

        run_id = self._id_generator()

        fp = None

        try:
            fp = self._file_system.open_text(run_id_file, mode=FileMode.WRITE)

            fp.write(run_id)
        except OSError as ex:
            raise_operational_system_error(ex)
        finally:
            if fp is not None:
                fp.close()

        return run_id


@runtime_checkable
class _WandbIdGenerator(Protocol):
    def __call__(self) -> str: ...


def _generate_wandb_id() -> str:
    return wandb.util.generate_id()
