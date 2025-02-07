# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from logging import Logger, getLogger
from typing import Any, Final, final


def get_log_writer(name: str | None = None) -> LogWriter:
    """Return a :class:`LogWriter` for the logger with the specified name."""
    return LogWriter(getLogger(name))


@final
class LogWriter:
    """Writes log messages using ``format()`` strings."""

    _NO_HIGHLIGHT: Final = {"highlighter": None}

    _logger: Logger

    def __init__(self, logger: Logger) -> None:
        """
        :param logger:
            The logger to write to.
        """
        self._logger = logger

    def debug(self, message: str, *args: Any) -> None:
        """Log a message with level ``DEBUG``."""
        self._write(logging.DEBUG, message, args)

    def info(self, message: str, *args: Any) -> None:
        """Log a message with level ``INFO``."""
        self._write(logging.INFO, message, args)

    def warning(self, message: str, *args: Any) -> None:
        """Log a message with level ``WARNING``."""
        self._write(logging.WARNING, message, args)

    def error(self, message: str, *args: Any, ex: BaseException | None = None) -> None:
        """Log a message with level ``ERROR``."""
        self._write(logging.ERROR, message, args, exc_info=ex or False)

    def exception(self, message: str, *args: Any) -> None:
        """Log a message with level ``ERROR``."""
        self._write(logging.ERROR, message, args, exc_info=True)

    def _write(
        self,
        level: int,
        message: str,
        args: tuple[Any, ...],
        exc_info: bool | BaseException = False,
    ) -> None:
        if args:
            if not self._logger.isEnabledFor(level):
                return

            message = str(message).format(*args)

        self._logger.log(level, message, exc_info=exc_info, extra=self._NO_HIGHLIGHT)

    def is_enabled_for(self, level: int) -> bool:
        """Return ``True`` if the writer is enabled for ``level``."""
        return self._logger.isEnabledFor(level)

    def is_enabled_for_debug(self) -> bool:
        return self._logger.isEnabledFor(logging.DEBUG)

    def is_enabled_for_info(self) -> bool:
        return self._logger.isEnabledFor(logging.INFO)

    def is_enabled_for_error(self) -> bool:
        return self._logger.isEnabledFor(logging.ERROR)


log = get_log_writer("fairseq2")


class LoggingSetupError(Exception):
    pass
