# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from logging import (
    DEBUG,
    INFO,
    FileHandler,
    Formatter,
    Handler,
    Logger,
    StreamHandler,
    getLogger,
)
from pathlib import Path
from typing import Any, List, Optional, final


def setup_logging(
    log_file: Optional[Path] = None,
    *,
    debug: bool = False,
    utc_time: bool = False,
    force: bool = False,
) -> None:
    """Set up logging for a training or eval job.

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

    handlers: List[Handler] = [StreamHandler()]  # Log to stderr.

    if log_file is not None:
        filename = log_file.name.format(rank=rank)

        if filename == log_file.name:
            raise ValueError(
                f"`log_file` must contain a 'rank' replacement field (i.e. {{rank}}) in its filename, but is '{log_file}' instead."
            )

        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as ex:
            raise RuntimeError(
                f"The log directory ({log_file.parent}) cannot be created. See nested exception for details."
            ) from ex

        handler = FileHandler(log_file.with_name(filename))

        handlers.append(handler)  # Log to file.

    fmt = f"[Rank {rank}] %(asctime)s %(levelname)s %(name)s - %(message)s"

    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=DEBUG if debug else INFO,
        handlers=handlers,
        format=fmt,
        datefmt=datefmt,
        force=force,
    )

    if utc_time:
        Formatter.converter = time.gmtime


@final
class LogWriter:
    """Writes log messages using ``format()`` strings."""

    _logger: Logger

    def __init__(self, logger: Logger) -> None:
        """
        :param logger:
            The logger to write to.
        """
        self._logger = logger

    def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log a message with level ``DEBUG``."""
        self._write(logging.DEBUG, msg, args, kwargs)

    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log a message with level ``INFO``."""
        self._write(logging.INFO, msg, args, kwargs)

    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log a message with level ``WARNING``."""
        self._write(logging.WARNING, msg, args, kwargs)

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log a message with level ``ERROR``."""
        self._write(logging.ERROR, msg, args, kwargs)

    def exception(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log a message with level ``ERROR``."""
        self._write(logging.ERROR, msg, args, kwargs, exc_info=True)

    def _write(
        self, level: int, msg: Any, args: Any, kwargs: Any, exc_info: bool = False
    ) -> None:
        if args or kwargs:
            if not self._logger.isEnabledFor(level):
                return

            msg = str(msg).format(*args, **kwargs)

        self._logger.log(level, msg, exc_info=exc_info)

    def is_enabled_for(self, level: int) -> bool:
        """Return ``True`` if a message of severity ``level`` would be processed
        by this writer."""
        return self._logger.isEnabledFor(level)


def get_log_writer(name: Optional[str] = None) -> LogWriter:
    """Return a :class:`LogWriter` for the logger with the specified name."""
    return LogWriter(getLogger(name))
