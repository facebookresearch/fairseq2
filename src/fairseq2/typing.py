# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import Field
from typing import Any, Callable, ClassVar, Dict, Final, Protocol, TypeVar

from torch import device, dtype
from typing_extensions import TypeAlias


class DataClass(Protocol):
    """Represents a data class object."""

    __dataclass_fields__: ClassVar[Dict[str, Field[Any]]]


F = TypeVar("F", bound=Callable[..., Any])


def override(f: F) -> F:
    """Indicate that the decorated member overrides an inherited virtual member."""
    return f


Device: TypeAlias = device

DataType: TypeAlias = dtype

CPU: Final = Device("cpu")

META: Final = Device("meta")
