# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
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
    CompositeMetricRecorder,
    MetricRecorder,
    NoopMetricRecorder,
    TensorBoardRecorder,
    WandbClient,
    WandbRecorder,
)
from fairseq2.recipe.config import CommonSection
from fairseq2.utils.structured import ValueConverter


@final
class MetricRecorderFactory:
    def __init__(
        self, gangs: Gangs, default_factory: Callable[[], CompositeMetricRecorder]
    ) -> None:
        self._gangs = gangs
        self._default_factory = default_factory

    def create(self) -> MetricRecorder:
        if self._gangs.root.rank != 0:
            return NoopMetricRecorder()

        return self._default_factory()


@final
class MaybeTensorBoardRecorderFactory:
    def __init__(
        self, section: CommonSection, factory: Callable[[], TensorBoardRecorder]
    ) -> None:
        self._section = section
        self._factory = factory

    def maybe_create(self) -> MetricRecorder | None:
        section = self._section.metric_recorders.tensorboard

        if not section.enabled:
            return None

        return self._factory()


@final
class MaybeWandbRecorderFactory:
    def __init__(
        self, section: CommonSection, factory: Callable[[], WandbRecorder]
    ) -> None:
        self._section = section
        self._factory = factory

    def maybe_create(self) -> MetricRecorder | None:
        section = self._section.metric_recorders.wandb

        if not section.enabled:
            return None

        if not _has_wandb:
            log.warning("wandb not found. Please install it with `pip install wandb`.")  # fmt: skip

            return None

        return self._factory()


@final
class WandbClientFactory:
    def __init__(
        self,
        section: CommonSection,
        output_dir: Path,
        config: object,
        value_converter: ValueConverter,
        run_id_manager: WandbRunIdManager,
        run_factory: WandbRunFactory,
    ) -> None:
        self._section = section
        self._output_dir = output_dir
        self._config = config
        self._value_converter = value_converter
        self._run_id_manager = run_id_manager
        self._run_factory = run_factory

    def create(self) -> WandbClient:
        unstructured_config = self._value_converter.unstructure(self._config)

        if not isinstance(unstructured_config, dict):
            unstructured_config = None

        run_id = self._run_id_manager.get_id()

        section = self._section.metric_recorders.wandb

        try:
            run = self._run_factory(
                entity=section.entity,
                project=section.project,
                dir=self._output_dir,
                id=run_id,
                name=section.run_name,
                config=unstructured_config,
                group=section.group,
                job_type=section.job_type,
                resume=section.resume_mode,
            )
        except (RuntimeError, ValueError) as ex:
            raise WandbInitializationError() from ex

        return WandbClient(run)


class WandbInitializationError(Exception):
    def __init__(self) -> None:
        super().__init__("Weights & Biases client cannot be initialized.")


@runtime_checkable
class WandbRunFactory(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def create_wandb_run(*args: Any, **kwargs: Any) -> Any:
    return wandb.init(*args, **kwargs)


class WandbRunIdManager(ABC):
    @abstractmethod
    def get_id(self) -> str: ...


@final
class StandardWandbRunIdManager(WandbRunIdManager):
    def __init__(
        self,
        section: CommonSection,
        file_system: FileSystem,
        id_generator: WandbIdGenerator,
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
class WandbIdGenerator(Protocol):
    def __call__(self) -> str: ...


def generate_wandb_id() -> str:
    return wandb.util.generate_id()
