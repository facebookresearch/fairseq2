# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Protocol, final


class MetricFormatter(Protocol):
    def __call__(self, value: object) -> str: ...


@dataclass
class MetricDescriptor:
    """
    Represents a description of a metric including high level name,
    name to display, and formatting
    """

    name: str
    display_name: str
    priority: int
    formatter: MetricFormatter
    log: bool = True
    higher_better: bool = False


NOOP_METRIC_DESCRIPTOR: Final = MetricDescriptor(
    name="", display_name="", priority=0, formatter=lambda value: "", log=False
)


@final
class MetricDescriptorRegistry:
    """
    Represents a way to store descriptors for multiple metrics in a composite metric
    """

    def __init__(self, descriptors: Iterable[MetricDescriptor]) -> None:
        self._descriptors = {d.name: d for d in descriptors}

    def maybe_get(self, name: str) -> MetricDescriptor | None:
        """
        Returns a metric descriptor if it exists
        """
        return self._descriptors.get(name)
