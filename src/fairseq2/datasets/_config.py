# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias


@dataclass
class StaticBatching:
    """Specifies batching where each batch has the same number of examples."""

    batch_size: int
    """The number of examples in each batch."""


@dataclass
class LengthBatching:
    """Specifies batching where each batch has a maximum number of elements."""

    max_num_elements: int
    """The maximum number of elements (e.g. tokens) in each batch."""


Batching: TypeAlias = StaticBatching | LengthBatching


class SyncMode(Enum):
    UNTIL_FIRST = 0
    """
    Stop data iteration on all ranks when one of the ranks reaches its end of
    data.
    """

    UNTIL_LAST = 1
    """
    Stop data iteration when all ranks reach their end of data; ranks that have
    already reached their end of data will return an empty list of batches.
    """


@dataclass(kw_only=True)
class DataReadOptions:
    batching: Batching = field(default_factory=lambda: StaticBatching(1))
    """The batching strategy for returned examples."""

    example_shuffle_window: int = 1
    """
    The size of the sliding window for shuffling examples. If ``1``, no
    shuffling is performed; if ``0``, true shuffling is performed by loading the
    entire dataset.
    """

    batch_shuffle_window: int = 1
    """
    The size of the sliding window for shuffling batches. If ``1``, no
    shuffling is performed; if ``0``, true shuffling is performed by loading the
    entire dataset.
    """

    drop_remainder: bool = False
    """
    If ``True``, drops the last set of batches if they have in total fewer
    examples than requested.
    """

    sync_batches: bool = True
    """
    If ``True``, ensures that each process in the gang reads the same number of
    batches. Typically used when the amount of data to be read can vary per
    process (e.g. due to unbalanced sharding or non-static batching) and it is
    critical for each process to iterate over the same number of batches (e.g.
    during training).
    """

    sync_mode: SyncMode = SyncMode.UNTIL_FIRST
    """
    The data synchronization mode among processes in the gang. Only effective if
    :attr:`sync_batches` is ``True``.
    """

    max_num_batches: int | None = None
    """The maximum number of batches to return."""

    num_accumulate: int = 1
    """
    The number of batches to accumulate in each iteration. Typically used with
    gradient accumulation during training.
    """

    num_prefetch: int = 1
    """The number of batches to prefetch in background."""

    seed: int = 2
    """The seed to initialize the random number generators used internally."""

    extras: MutableMapping[str, object] = field(default_factory=dict)
    """The reader-specific extra options."""
