# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from rich import get_console as get_rich_console
from rich.console import Console

_console: Optional[Console] = None


def get_console() -> Console:
    """Return the ``stdout`` Rich console."""
    global _console

    if _console is None:
        _console = get_rich_console()

    return _console


def set_console(console: Console) -> None:
    """Set the ``stdout`` Rich console."""
    global _console

    _console = console


_error_console: Optional[Console] = None


def get_error_console() -> Console:
    """Return the ``stderr`` Rich console."""
    global _error_console

    if _error_console is None:
        _error_console = Console(stderr=True, highlight=False)

    return _error_console


def set_error_console(console: Console) -> None:
    """Get the ``stderr`` Rich console."""
    global _error_console

    _error_console = console
