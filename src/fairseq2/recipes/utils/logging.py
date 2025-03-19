# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import MutableMapping
from logging import FileHandler, Formatter, Logger
from pathlib import Path
from typing import TYPE_CHECKING, final

from fairseq2n import DOC_MODE

from fairseq2.logging import LoggingSetupError
from fairseq2.utils.env import InvalidEnvironmentVariableError, get_rank
from fairseq2.utils.file import FileSystem


@final
class DistributedLoggingInitializer:
    _logger: Logger
    _env: MutableMapping[str, str]
    _file_system: FileSystem

    def __init__(
        self, logger: Logger, env: MutableMapping[str, str], file_system: FileSystem
    ) -> None:
        self._logger = logger
        self._env = env
        self._file_system = file_system

    def initialize(self, output_dir: Path) -> None:
        try:
            rank = get_rank(self._env)
        except InvalidEnvironmentVariableError as ex:
            raise LoggingSetupError(
                "The rank of the process cannot be determined. See the nested exception for details."
            ) from ex

        if rank != 0:
            for handler in self._logger.handlers[:]:
                self._logger.removeHandler(handler)

        log_file = output_dir.joinpath(f"logs/rank_{rank}.log")

        try:
            self._file_system.make_directory(log_file.parent)
        except OSError as ex:
            raise LoggingSetupError(
                f"The '{log_file}' log file cannot be created. See the nested exception for details."
            ) from ex

        handler = FileHandler(log_file)

        handler.setFormatter(
            Formatter(f"[Rank {rank}] %(asctime)s %(levelname)s %(name)s - %(message)s")
        )

        self._logger.addHandler(handler)

        self._setup_aten_logging(log_file)
        self._setup_nccl_logging(log_file)

    def _setup_aten_logging(self, log_file: Path) -> None:
        if "TORCH_CPP_LOG_LEVEL" in self._env:
            return

        aten_log_file = log_file.parent.joinpath("aten", log_file.name)

        try:
            self._file_system.make_directory(aten_log_file.parent)
        except OSError as ex:
            raise LoggingSetupError(
                f"The '{aten_log_file.parent}' ATen log directory cannot be created. See the nested exception for details."
            ) from ex

        _enable_aten_logging(aten_log_file)

        # This variable has no effect at this point; set for completeness.
        self._env["TORCH_CPP_LOG_LEVEL"] = "INFO"

    def _setup_nccl_logging(self, log_file: Path) -> None:
        if "NCCL_DEBUG" in self._env:
            return

        nccl_log_file = log_file.parent.joinpath("nccl", log_file.name)

        try:
            self._file_system.make_directory(nccl_log_file.parent)
        except OSError as ex:
            raise LoggingSetupError(
                f"The '{nccl_log_file.parent}' NCCL log directory cannot be created. See the nested exception for details."
            ) from ex

        self._env["NCCL_DEBUG"] = "INFO"
        self._env["NCCL_DEBUG_FILE"] = str(nccl_log_file)


if TYPE_CHECKING or DOC_MODE:

    def _enable_aten_logging(log_file: Path) -> Path: ...

else:
    from fairseq2n.bindings import _enable_aten_logging
