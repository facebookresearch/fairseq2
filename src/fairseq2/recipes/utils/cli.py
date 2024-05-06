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


def create_rich_progress() -> Progress:
    """Create a :class:`Progress` instance to report training progress."""
    return Progress(
        BarColumn(), MofNCompleteColumn(), TaskProgressColumn(), TimeRemainingColumn()
    )
