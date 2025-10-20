# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from logging import INFO, Formatter, NullHandler, getLogger
from typing import Any, final

from rich import get_console as get_rich_console
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.text import Text
from typing_extensions import Self, override

from fairseq2.utils.env import StandardEnvironment, get_rank
from fairseq2.utils.progress import ProgressReporter, ProgressTask
from fairseq2.world_info import WorldInfo

_console: Console | None = None


def get_console() -> Console:
    global _console

    if _console is None:
        _console = get_rich_console()

    return _console


def set_console(console: Console) -> None:
    global _console

    _console = console


_error_console: Console | None = None


def get_error_console() -> Console:
    global _error_console

    if _error_console is None:
        _error_console = Console(stderr=True, highlight=False)

    return _error_console


def set_error_console(console: Console) -> None:
    global _error_console

    _error_console = console


@final
class RichProgressReporter(ProgressReporter):
    def __init__(
        self,
        console: Console,
        world_info: WorldInfo,
        columns: Sequence[ProgressColumn] | None = None,
    ) -> None:
        disable = world_info.rank != 0

        if columns is None:
            columns = [
                TextColumn("{task.description}:", style="progress.description"),
                BarColumn(),
                BasicMofNCompleteColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            ]

        self._progress = Progress(
            *columns, transient=True, console=console, disable=disable
        )

    @override
    def create_task(
        self, name: str, total: int | None, completed: int = 0, *, start: bool = True
    ) -> ProgressTask:
        task_id = self._progress.add_task(
            name, start=start, total=total, completed=completed
        )

        return RichProgressTask(self._progress, task_id)

    @override
    def __enter__(self) -> Self:
        self._progress.__enter__()

        return self

    @override
    def __exit__(self, *ex: Any) -> None:
        self._progress.__exit__(*ex)


@final
class RichProgressTask(ProgressTask):
    def __init__(self, progress: Progress, task_id: TaskID) -> None:
        self._progress = progress
        self._task_id = task_id

    @override
    def start(self) -> None:
        self._progress.start_task(self._task_id)

    @override
    def step(self, value: int = 1) -> None:
        self._progress.update(self._task_id, advance=value)

    @override
    def close(self) -> None:
        self._progress.remove_task(self._task_id)


class BasicMofNCompleteColumn(ProgressColumn):
    @override
    def render(self, task: Task) -> Text:
        if task.total is None:
            if not task.started:
                return Text()

            s = f"{task.completed:5d}"
        else:
            s = f"{task.completed:5d}/{task.total}"

        return Text(s, style="progress.download")


def create_rich_download_progress_columns() -> list[ProgressColumn | str]:
    return [
        TextColumn("{task.description}:", style="progress.description"),
        TextColumn("{task.percentage:>4.0f}%", style="progress.percentage"),
        BarColumn(),
        DownloadColumn(binary_units=True),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ]


def configure_rich_logging() -> None:
    logger = getLogger("fairseq2")

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

        handler.close()

    env = StandardEnvironment()

    rank = get_rank(env)
    if rank is None or rank == 0:
        console = get_error_console()

        handler = RichHandler(console=console, show_path=False, keywords=[])

        console_formatter = Formatter(
            fmt="%(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        handler.setFormatter(console_formatter)
    else:
        handler = NullHandler()

    logger.addHandler(handler)

    logger.setLevel(INFO)

    logger.propagate = False
