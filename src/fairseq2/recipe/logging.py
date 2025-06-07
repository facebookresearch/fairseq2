# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from logging import DEBUG, FileHandler, Formatter
from pathlib import Path
from typing import TYPE_CHECKING, final

from fairseq2n import DOC_MODE
from typing_extensions import override

from fairseq2.error import raise_operational_system_error
from fairseq2.file_system import FileSystem
from fairseq2.logging import log
from fairseq2.recipe.config import CommonSection
from fairseq2.utils.env import Environment
from fairseq2.world_info import WorldInfo


class DistributedLogConfigurer(ABC):
    @abstractmethod
    def configure(self, output_dir: Path) -> None: ...


@final
class StandardDistributedLogConfigurer(DistributedLogConfigurer):
    def __init__(
        self,
        section: CommonSection,
        env: Environment,
        world_info: WorldInfo,
        file_system: FileSystem,
    ) -> None:
        self._section = section
        self._env = env
        self._world_info = world_info
        self._file_system = file_system

    @override
    def configure(self, output_dir: Path) -> None:
        self._configure_logger(output_dir)

        self._configure_aten_logging(output_dir)

        self._configure_nccl_logging(output_dir)

        log.info("Log files are stored under {}.", output_dir)

    def _configure_logger(self, output_dir: Path) -> None:
        logger = log.logger

        rank = self._world_info.rank

        if rank != 0:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

                handler.close()

        log_file = output_dir.joinpath(f"logs/rank_{rank}.log")

        try:
            self._file_system.make_directory(log_file.parent)
        except OSError as ex:
            raise_operational_system_error(ex)

        try:
            handler = FileHandler(log_file)
        except OSError as ex:
            raise_operational_system_error(ex)

        fmt = f"[Rank {rank}] %(asctime)s %(levelname)s %(name)s - %(message)s"

        file_formatter = Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

        handler.setFormatter(file_formatter)

        logger.addHandler(handler)

        if self._section.debug:
            logger.setLevel(DEBUG)

    def _configure_aten_logging(self, output_dir: Path) -> None:
        if self._env.has("TORCH_CPP_LOG_LEVEL"):
            return

        log_file = output_dir.joinpath(f"logs/aten/rank_{self._world_info.rank}.log")

        try:
            self._file_system.make_directory(log_file.parent)
        except OSError as ex:
            raise_operational_system_error(ex)

        _enable_aten_logging(log_file)

        # This variable has no effect at this point; set for completeness.
        self._env.set("TORCH_CPP_LOG_LEVEL", "INFO")

    def _configure_nccl_logging(self, output_dir: Path) -> None:
        if self._env.has("NCCL_DEBUG"):
            return

        log_file = output_dir.joinpath(f"logs/nccl/rank_{self._world_info.rank}.log")

        try:
            self._file_system.make_directory(log_file.parent)
        except OSError as ex:
            raise_operational_system_error(ex)

        self._env.set("NCCL_DEBUG", "INFO")
        self._env.set("NCCL_DEBUG_FILE", str(log_file))


if TYPE_CHECKING or DOC_MODE:

    def _enable_aten_logging(log_file: Path) -> None: ...

else:
    from fairseq2n.bindings import _enable_aten_logging
