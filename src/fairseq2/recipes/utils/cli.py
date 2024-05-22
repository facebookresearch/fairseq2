# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from fairseq2.gang import get_rank
from fairseq2.recipes.logging import console


def create_rich_progress() -> Progress:
    """Create a :class:`Progress` instance to report job progress."""
    columns = [
        BarColumn(), MofNCompleteColumn(), TaskProgressColumn(), TimeRemainingColumn()  # fmt: skip
    ]

    rank = get_rank()

    return Progress(
        *columns, auto_refresh=False, transient=True, console=console, disable=rank != 0
    )
