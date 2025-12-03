# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from logging import INFO, Formatter, Logger, NullHandler, StreamHandler, getLogger
from typing import Any, Final, final

from fairseq2.error import FormatError
from fairseq2.utils.env import StandardEnvironment, maybe_get_rank


@final
class LogWriter:
    """Writes log messages using ``format()`` strings."""

    _NO_HIGHLIGHT: Final = {"highlighter": None}

    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    def debug(
        self, message: str, *args: Any, exc: BaseException | None = None, **kwargs: Any
    ) -> None:
        self._log(logging.DEBUG, message, args, kwargs, exc or False)

    def info(
        self, message: str, *args: Any, exc: BaseException | None = None, **kwargs: Any
    ) -> None:
        self._log(logging.INFO, message, args, kwargs, exc or False)

    def warning(
        self, message: str, *args: Any, exc: BaseException | None = None, **kwargs: Any
    ) -> None:
        self._log(logging.WARNING, message, args, kwargs, exc or False)

    def error(
        self, message: str, *args: Any, exc: BaseException | None = None, **kwargs: Any
    ) -> None:
        self._log(logging.ERROR, message, args, kwargs, exc or False)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.ERROR, message, args, kwargs, exc_info=True)

    def _log(
        self,
        level: int,
        message: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        exc_info: bool | BaseException = False,
    ) -> None:
        if args or kwargs:
            if not self._logger.isEnabledFor(level):
                return

            message = message.format(*args, **kwargs)

        self._logger.log(level, message, exc_info=exc_info, extra=self._NO_HIGHLIGHT)

    def is_enabled_for(self, level: int) -> bool:
        return self._logger.isEnabledFor(level)

    def is_enabled_for_debug(self) -> bool:
        return self._logger.isEnabledFor(logging.DEBUG)

    def is_enabled_for_info(self) -> bool:
        return self._logger.isEnabledFor(logging.INFO)


def get_log_writer(name: str | None = None) -> LogWriter:
    """Returns the :class:`LogWriter` for the specified name."""
    return LogWriter(getLogger(name))


log = get_log_writer("fairseq2")


def configure_logging(no_rich: bool = False) -> None:
    """
    :raises EnvironmentVariableError:
    """
    logger = getLogger()

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

        handler.close()

    env = StandardEnvironment()

    try:
        rank = maybe_get_rank(env)
    except FormatError:
        rank = None

    if rank is None or rank == 0:
        datefmt = "%Y-%m-%d %H:%M:%S"

        if no_rich:
            handler = StreamHandler()

            console_formatter = Formatter(
                "%(asctime)s %(levelname)s: %(name)s - %(message)s", datefmt
            )
        else:
            from rich.logging import RichHandler

            from fairseq2.utils.rich import get_error_console

            console = get_error_console()

            handler = RichHandler(console=console, show_path=False, keywords=[])

            console_formatter = Formatter("%(name)s - %(message)s", datefmt)

        handler.setFormatter(console_formatter)
    else:
        handler = NullHandler()

    logger.addHandler(handler)

    logger.setLevel(INFO)

    logger.propagate = False
