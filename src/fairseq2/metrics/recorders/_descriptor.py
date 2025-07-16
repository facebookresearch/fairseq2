# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Final, Protocol

from torch import Tensor


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


def format_as_int(value: object, *, postfix: str | None = None) -> str:
    """Format metric ``value`` as integer."""
    if isinstance(value, int):
        i = value
    elif isinstance(value, (str, Tensor, float)):
        try:
            i = int(value)
        except ValueError:
            return f"{value}"
    else:
        return f"{value}"

    s = "<1" if i == 0 and isinstance(value, float) else f"{i:,}"

    if postfix:
        s += postfix

    return s


format_as_seconds = partial(format_as_int, postfix="s")
"""Format metric ``value`` as duration in seconds."""


def format_as_float(value: object, *, postfix: str | None = None) -> str:
    """Format metric ``value`` as float."""
    if isinstance(value, float):
        f = value
    elif isinstance(value, (str, Tensor, int)):
        try:
            f = float(value)
        except ValueError:
            return f"{value}"
    else:
        return f"{value}"

    s = f"{f:g}"

    if postfix:
        s += postfix

    return s


def format_as_percentage(value: object) -> str:
    """Format metric ``value`` as percentage."""
    if isinstance(value, float):
        f = value
    elif isinstance(value, (str, Tensor, int)):
        try:
            f = float(value)
        except ValueError:
            return f"{value}"
    else:
        return f"{value}"

    i = math.ceil(f * 100)

    return f"{i}%"


_UNITS: Final = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]


def format_as_byte_size(value: object) -> str:
    """Format metric ``value`` in byte units."""
    if isinstance(value, float):
        size = value
    elif isinstance(value, (str, Tensor, int)):
        try:
            size = float(value)
        except ValueError:
            return f"{value}"
    else:
        return f"{value}"

    unit_idx = 0

    if not math.isfinite(size) or size <= 0.0:
        return "0 B"

    while size >= 1024:
        size /= 1024

        unit_idx += 1

    try:
        return f"{size:.2f} {_UNITS[unit_idx]}"
    except IndexError:
        return "TOO BIG"
