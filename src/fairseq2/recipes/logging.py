# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
from logging import DEBUG, INFO, FileHandler, Formatter, Handler, NullHandler, getLogger
from pathlib import Path
from typing import TYPE_CHECKING, List

from fairseq2n import DOC_MODE
from rich.logging import RichHandler

from fairseq2.console import get_error_console


def setup_basic_logging(*, debug: bool = False, utc_time: bool = False) -> None:
    """Set up logging for a command line program.

    :param debug:
        If ``True``, sets the log level to ``DEBUG``; otherwise, to ``INFO``.
    :param utc_time:
        If ``True``, logs dates and times in UTC.
    """
    from fairseq2.gang import get_rank  # Avoid circular import.

    rank = get_rank()

    _do_setup_logging(rank, debug, utc_time)

    if rank != 0:
        getLogger().addHandler(NullHandler())


def setup_logging(
    log_file: Path, *, debug: bool = False, utc_time: bool = False, force: bool = False
) -> None:
    """Set up logging for a distributed job.

    :param log_file:
        The file to which logs will be written. Must have a 'rank' replacement
        field; for example '/path/to/train_{rank}.log'.
    :param debug:
        If ``True``, sets the log level to ``DEBUG``; otherwise, to ``INFO``.
    :param utc_time:
        If ``True``, logs dates and times in UTC.
    :param force:
        If ``True``, overwrites existing ATen and NCCL log configurations.
    """
    from fairseq2.gang import get_rank  # Avoid circular import.

    rank = get_rank()

    filename = log_file.name.format(rank=rank)

    if filename == log_file.name:
        raise ValueError(
            f"`log_file` must contain a 'rank' replacement field (i.e. {{rank}}) in its filename, but is '{log_file}' instead."
        )

    log_file = log_file.with_name(filename)

    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as ex:
        raise RuntimeError(
            f"The log directory ({log_file.parent}) cannot be created. See nested exception for details."
        ) from ex

    _do_setup_logging(rank, debug, utc_time)

    handler = FileHandler(log_file)

    fmt = Formatter(f"[Rank {rank}] %(asctime)s %(levelname)s %(name)s - %(message)s")

    handler.setFormatter(fmt)

    getLogger().addHandler(handler)

    _setup_aten_logging(log_file, force)
    _setup_nccl_logging(log_file, force)


def _do_setup_logging(rank: int, debug: bool = False, utc_time: bool = False) -> None:
    if utc_time:
        Formatter.converter = time.gmtime

    handlers: List[Handler] = []

    if rank == 0:
        console = get_error_console()

        handler = RichHandler(console=console, show_path=False, keywords=[])

        fmt = Formatter("%(name)s - %(message)s")

        handler.setFormatter(fmt)

        handlers.append(handler)

    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=DEBUG if debug else INFO, handlers=handlers, datefmt=datefmt, force=True
    )


def _setup_aten_logging(log_file: Path, force: bool) -> None:
    if "TORCH_CPP_LOG_LEVEL" in os.environ and not force:
        return

    aten_log_file = log_file.parent.joinpath("aten", log_file.name)

    try:
        aten_log_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as ex:
        raise RuntimeError(
            f"The ATen log directory ({aten_log_file.parent}) cannot be created. See nested exception for details."
        ) from ex

    _enable_aten_logging(aten_log_file)

    # This variable has no effect at this point. We set it for completeness.
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"


if TYPE_CHECKING or DOC_MODE:

    def _enable_aten_logging(log_file: Path) -> Path:
        ...

else:
    from fairseq2n.bindings import _enable_aten_logging


def _setup_nccl_logging(log_file: Path, force: bool) -> None:
    if "NCCL_DEBUG" in os.environ and not force:
        return

    nccl_log_file = log_file.parent.joinpath("nccl", log_file.name)

    try:
        nccl_log_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as ex:
        raise RuntimeError(
            f"The NCCL log directory ({nccl_log_file.parent}) cannot be created. See nested exception for details."
        ) from ex

    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_FILE"] = str(nccl_log_file)
