# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from logging import Logger, getLogger
from typing import Any, Final, final


@final
class LogWriter:
    """Writes log messages using ``format()`` strings."""

    _NO_HIGHLIGHT: Final = {"highlighter": None}

    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    def debug(self, message: str, *args: Any) -> None:
        self._log(logging.DEBUG, message, args)

    def info(self, message: str, *args: Any) -> None:
        self._log(logging.INFO, message, args)

    def warning(self, message: str, *args: Any) -> None:
        self._log(logging.WARNING, message, args)

    def error(self, message: str, *args: Any) -> None:
        self._log(logging.ERROR, message, args)

    def exception(self, message: str, *args: Any) -> None:
        self._log(logging.ERROR, message, args, exc_info=True)

    def _log(
        self, level: int, message: str, args: tuple[Any, ...], exc_info: bool = False
    ) -> None:
        if args:
            if not self._logger.isEnabledFor(level):
                return

            message = str(message).format(*args)

        self._logger.log(level, message, exc_info=exc_info, extra=self._NO_HIGHLIGHT)

    def is_enabled_for(self, level: int) -> bool:
        return self._logger.isEnabledFor(level)

    def is_enabled_for_debug(self) -> bool:
        return self._logger.isEnabledFor(logging.DEBUG)

    def is_enabled_for_info(self) -> bool:
        return self._logger.isEnabledFor(logging.INFO)

    @property
    def logger(self) -> Logger:
        return self._logger


def get_log_writer(name: str | None = None) -> LogWriter:
    """Return a :class:`LogWriter` for the logger with the specified name."""
    return LogWriter(getLogger(name))


log = get_log_writer("fairseq2")
