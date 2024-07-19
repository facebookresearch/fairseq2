# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from logging import Logger, getLogger
from typing import Any, Final, Optional, final


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

    def debug(
        self, msg: Any, *args: Any, highlight: bool = False, **kwargs: Any
    ) -> None:
        """Log a message with level ``DEBUG``."""
        self._write(logging.DEBUG, msg, args, kwargs, highlight)

    def info(
        self, msg: Any, *args: Any, highlight: bool = False, **kwargs: Any
    ) -> None:
        """Log a message with level ``INFO``."""
        self._write(logging.INFO, msg, args, kwargs, highlight)

    def warning(
        self, msg: Any, *args: Any, highlight: bool = False, **kwargs: Any
    ) -> None:
        """Log a message with level ``WARNING``."""
        self._write(logging.WARNING, msg, args, kwargs, highlight)

    def error(
        self, msg: Any, *args: Any, highlight: bool = False, **kwargs: Any
    ) -> None:
        """Log a message with level ``ERROR``."""
        self._write(logging.ERROR, msg, args, kwargs, highlight)

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
        """Return ``True`` if the writer is enabled for ``level``."""
        return self._logger.isEnabledFor(level)

    def is_enabled_for_debug(self) -> bool:
        """Return ``True`` if the writer is enabled for ``logging.DEBUG``."""
        return self._logger.isEnabledFor(logging.DEBUG)

    def is_enabled_for_info(self) -> bool:
        """Return ``True`` if the writer is enabled for ``loggig.INFO``."""
        return self._logger.isEnabledFor(logging.INFO)


def get_log_writer(name: Optional[str] = None) -> LogWriter:
    """Return a :class:`LogWriter` for the logger with the specified name."""
    return LogWriter(getLogger(name))
