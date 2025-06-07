# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class MetricFormatter(Protocol):
    def __call__(self, value: object) -> str: ...


@dataclass
class MetricDescriptor:
    name: str
    display_name: str
    priority: int
    formatter: MetricFormatter
    log: bool = True
    higher_better: bool = False


class UnknownMetricDescriptorError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known metric.")

        self.name = name
