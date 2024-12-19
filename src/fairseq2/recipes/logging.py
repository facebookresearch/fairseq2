# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from logging import DEBUG, INFO, FileHandler, Formatter, Handler, NullHandler, getLogger
from pathlib import Path
from typing import TYPE_CHECKING, final

from fairseq2n import DOC_MODE
from rich.logging import RichHandler
from typing_extensions import override

from fairseq2.error import SetupError
from fairseq2.gang import get_rank
from fairseq2.recipes.console import get_error_console


def setup_basic_logging(*, debug: bool = False, utc_time: bool = False) -> None:
    rank = get_rank()

    _setup_core_logging(rank, debug, utc_time)

    if rank != 0:
        getLogger().addHandler(NullHandler())


class LoggingInitializer(ABC):
    @abstractmethod
    def initialize(
        self, log_file: Path, *, debug: bool = False, utc_time: bool = False
    ) -> None:
        ...


@final
class DistributedLoggingInitializer(LoggingInitializer):
    @override
    def initialize(
        self, log_file: Path, *, debug: bool = False, utc_time: bool = False
    ) -> None:
        rank = get_rank()

        filename = log_file.name.format(rank=rank)

        if filename == log_file.name:
            raise ValueError(
                f"`log_file` must have a 'rank' replacement field (i.e. {{rank}}) in its filename, but is '{log_file}' instead."
            )

        log_file = log_file.with_name(filename)

        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as ex:
            raise SetupError(
                f"The '{log_file}' log file cannot be created. See the nested exception for details."
            ) from ex

        _setup_core_logging(rank, debug, utc_time)

        handler = FileHandler(log_file)

        handler.setFormatter(
            Formatter(f"[Rank {rank}] %(asctime)s %(levelname)s %(name)s - %(message)s")
        )

        getLogger().addHandler(handler)

        _setup_aten_logging(log_file)
        _setup_nccl_logging(log_file)


def _setup_core_logging(rank: int, debug: bool = False, utc_time: bool = False) -> None:
    level = DEBUG if debug else INFO

    if utc_time:
        Formatter.converter = time.gmtime

    handlers: list[Handler] = []

    if rank == 0:
        console = get_error_console()

        handler = RichHandler(console=console, show_path=False, keywords=[])

        handler.setFormatter(Formatter("%(name)s - %(message)s"))

        handlers.append(handler)

    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(level=level, handlers=handlers, datefmt=datefmt, force=True)


def _setup_aten_logging(log_file: Path) -> None:
    if "TORCH_CPP_LOG_LEVEL" in os.environ:
        return

    aten_log_file = log_file.parent.joinpath("aten", log_file.name)

    aten_log_file.parent.mkdir(parents=True, exist_ok=True)

    _enable_aten_logging(aten_log_file)

    # This variable has no effect at this point; set for completeness.
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"


if TYPE_CHECKING or DOC_MODE:

    def _enable_aten_logging(log_file: Path) -> Path:
        ...

else:
    from fairseq2n.bindings import _enable_aten_logging


def _setup_nccl_logging(log_file: Path) -> None:
    if "NCCL_DEBUG" in os.environ:
        return

    nccl_log_file = log_file.parent.joinpath("nccl", log_file.name)

    nccl_log_file.parent.mkdir(parents=True, exist_ok=True)

    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_FILE"] = str(nccl_log_file)
