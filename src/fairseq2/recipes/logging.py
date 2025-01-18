# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from logging import FileHandler, Formatter, getLogger
from pathlib import Path
from typing import TYPE_CHECKING, final

from fairseq2n import DOC_MODE
from typing_extensions import override

from fairseq2.error import SetupError
from fairseq2.gang import get_rank
from fairseq2.utils.file import FileSystem


class LoggingInitializer(ABC):
    @abstractmethod
    def initialize(self, log_file: Path) -> None:
        ...


@final
class DistributedLoggingInitializer(LoggingInitializer):
    _file_system: FileSystem

    def __init__(self, file_system: FileSystem) -> None:
        self._file_system = file_system

    @override
    def initialize(self, log_file: Path) -> None:
        rank = get_rank()

        logger = getLogger()

        if rank != 0:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

        filename = log_file.name.format(rank=rank)

        if filename == log_file.name:
            raise ValueError(
                f"`log_file` must have a 'rank' replacement field (i.e. {{rank}}) in its filename, but is '{log_file}' instead."
            )

        log_file = log_file.with_name(filename)

        try:
            self._file_system.make_directory(log_file.parent)
        except OSError as ex:
            raise SetupError(
                f"The '{log_file}' log file cannot be created. See the nested exception for details."
            ) from ex

        handler = FileHandler(log_file)

        handler.setFormatter(
            Formatter(f"[Rank {rank}] %(asctime)s %(levelname)s %(name)s - %(message)s")
        )

        logger.addHandler(handler)

        self._setup_aten_logging(log_file)
        self._setup_nccl_logging(log_file)

    def _setup_aten_logging(self, log_file: Path) -> None:
        if "TORCH_CPP_LOG_LEVEL" in os.environ:
            return

        aten_log_file = log_file.parent.joinpath("aten", log_file.name)

        self._file_system.make_directory(aten_log_file.parent)

        _enable_aten_logging(aten_log_file)

        # This variable has no effect at this point; set for completeness.
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"

    def _setup_nccl_logging(self, log_file: Path) -> None:
        if "NCCL_DEBUG" in os.environ:
            return

        nccl_log_file = log_file.parent.joinpath("nccl", log_file.name)

        self._file_system.make_directory(nccl_log_file.parent)

        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_DEBUG_FILE"] = str(nccl_log_file)


if TYPE_CHECKING or DOC_MODE:

    def _enable_aten_logging(log_file: Path) -> Path:
        ...

else:
    from fairseq2n.bindings import _enable_aten_logging
