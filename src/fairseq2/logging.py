# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
from logging import DEBUG, INFO, FileHandler, Formatter, Handler, Logger, getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, List, Optional, Set, final

from fairseq2n import DOC_MODE
from rich.logging import RichHandler


def setup_logging(
    log_file: Path,
    *,
    debug: bool = False,
    utc_time: bool = False,
    force: bool = False,
) -> None:
    """Set up logging for a training or evaluation job.

    :param log_file:
        The file to which logs will be written. Must have a 'rank' replacement
        field; for example '/path/to/train_{rank}.log'.
    :param debug:
        If ``True``, sets the log level to ``DEBUG``; otherwise, to ``INFO``.
    :param utc_time:
        If ``True``, logs dates and times in UTC.
    :param force:
        If ``True``, overwrites existing log configuration.
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

    if utc_time:
        Formatter.converter = time.gmtime

    file_handler = FileHandler(log_file)

    formatter = Formatter(
        f"[Rank {rank}] %(asctime)s %(levelname)s %(name)s - %(message)s"
    )

    file_handler.setFormatter(formatter)

    handlers: List[Handler] = [file_handler]

    if rank == 0:
        rich_handler = RichHandler(show_path=False, keywords=[])

        formatter = Formatter("%(name)s - %(message)s")

        rich_handler.setFormatter(formatter)

        handlers.append(rich_handler)

    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=DEBUG if debug else INFO, handlers=handlers, datefmt=datefmt, force=force
    )

    _setup_aten_logging(log_file, force)

    _setup_nccl_logging(log_file, force)


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


@final
class LogWriter:
    """Writes log messages using ``format()`` strings."""

    _NO_HIGHLIGHT: Final = {"highlighter": None}

    _logger: Logger
    _once_messages: Set[str]

    def __init__(self, logger: Logger) -> None:
        """
        :param logger:
            The logger to write to.
        """
        self._logger = logger

        self._once_messages = set()

    def debug(
        self, msg: Any, *args: Any, highlight: bool = False, **kwargs: Any
    ) -> None:
        """Log a message with level ``DEBUG``."""
        self._write(logging.DEBUG, msg, args, kwargs, highlight)

    def debug_once(
        self, msg: Any, *args: Any, highlight: bool = False, **kwargs: Any
    ) -> None:
        """Log a message only once with level ``DEBUG``."""
        if msg in self._once_messages:
            return

        self._write(logging.DEBUG, msg, args, kwargs, highlight)

        self._once_messages.add(msg)

    def info(
        self, msg: Any, *args: Any, highlight: bool = False, **kwargs: Any
    ) -> None:
        """Log a message with level ``INFO``."""
        self._write(logging.INFO, msg, args, kwargs, highlight)

    def info_once(
        self, msg: Any, *args: Any, highlight: bool = False, **kwargs: Any
    ) -> None:
        """Log a message only once with level ``INFO``."""
        if msg in self._once_messages:
            return

        self._write(logging.INFO, msg, args, kwargs, highlight)

        self._once_messages.add(msg)

    def warning(
        self, msg: Any, *args: Any, highlight: bool = False, **kwargs: Any
    ) -> None:
        """Log a message with level ``WARNING``."""
        self._write(logging.WARNING, msg, args, kwargs, highlight)

    def warning_once(
        self, msg: Any, *args: Any, highlight: bool = False, **kwargs: Any
    ) -> None:
        """Log a message only once with level ``WARNING``."""
        if msg in self._once_messages:
            return

        self._write(logging.WARNING, msg, args, kwargs, highlight)

        self._once_messages.add(msg)

    def error(
        self, msg: Any, *args: Any, highlight: bool = False, **kwargs: Any
    ) -> None:
        """Log a message with level ``ERROR``."""
        self._write(logging.ERROR, msg, args, kwargs, highlight)

    def error_once(
        self, msg: Any, *args: Any, highlight: bool = False, **kwargs: Any
    ) -> None:
        """Log a message only once with level ``ERROR``."""
        if msg in self._once_messages:
            return

        self._write(logging.ERROR, msg, args, kwargs, highlight)

        self._once_messages.add(msg)

    def exception(
        self, msg: Any, *args: Any, highlight: bool = False, **kwargs: Any
    ) -> None:
        """Log a message with level ``ERROR``."""
        self._write(logging.ERROR, msg, args, kwargs, highlight, exc_info=True)

    def _write(
        self,
        level: int,
        msg: Any,
        args: Any,
        kwargs: Any,
        highlight: bool,
        exc_info: bool = False,
    ) -> None:
        if args or kwargs:
            if not self._logger.isEnabledFor(level):
                return

            msg = str(msg).format(*args, **kwargs)

        extra = None if highlight else self._NO_HIGHLIGHT

        self._logger.log(level, msg, extra=extra, exc_info=exc_info)

    def is_enabled_for(self, level: int) -> bool:
        """Return ``True`` if a message of severity ``level`` would be processed
        by this writer."""
        return self._logger.isEnabledFor(level)


def get_log_writer(name: Optional[str] = None) -> LogWriter:
    """Return a :class:`LogWriter` for the logger with the specified name."""
    return LogWriter(getLogger(name))
