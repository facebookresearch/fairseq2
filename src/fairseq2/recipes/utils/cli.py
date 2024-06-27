# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from fairseq2.console import get_error_console
from fairseq2.gang import get_rank
from fairseq2.typing import override


def create_rich_progress() -> Progress:
    """Create a :class:`Progress` instance to report job progress."""
    console = get_error_console()

    columns = [
        TextColumn("{task.description}:"),
        BarColumn(),
        BasicMofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ]

    rank = get_rank()

    return Progress(*columns, transient=True, console=console, disable=rank != 0)


class BasicMofNCompleteColumn(ProgressColumn):
    @override
    def render(self, task: Task) -> Text:
        if task.total is None:
            s = f"{task.completed:5d}"
        else:
            s = f"{task.completed:5d}/{task.total}"

        return Text(s, style="progress.download")
