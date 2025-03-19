# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, final

from rich import get_console as get_rich_console
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from typing_extensions import Self, override

from fairseq2.utils.progress import ProgressReporter, ProgressTask

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
    def __init__(self, console: Console, rank: int) -> None:
        columns = [
            TextColumn("{task.description}:"),
            BarColumn(),
            BasicMofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ]

        self._progress = Progress(
            *columns, transient=True, console=console, disable=rank != 0
        )

    @override
    def create_task(
        self, name: str, total: int | None, completed: int = 0
    ) -> ProgressTask:
        task_id = self._progress.add_task(name, total=total, completed=completed)

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
    _progress: Progress
    _task_id: TaskID

    def __init__(self, progress: Progress, task_id: TaskID) -> None:
        self._progress = progress
        self._task_id = task_id

    @override
    def step(self, value: int) -> None:
        self._progress.update(self._task_id, advance=value)

    @override
    def close(self) -> None:
        self._progress.remove_task(self._task_id)


class BasicMofNCompleteColumn(ProgressColumn):
    @override
    def render(self, task: Task) -> Text:
        if task.total is None:
            s = f"{task.completed:5d}"
        else:
            s = f"{task.completed:5d}/{task.total}"

        return Text(s, style="progress.download")


def create_rich_progress_reporter(rank: int) -> ProgressReporter:
    console = get_error_console()

    return RichProgressReporter(console, rank)
