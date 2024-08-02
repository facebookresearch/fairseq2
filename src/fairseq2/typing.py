# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import Field, is_dataclass
from typing import Any, Callable, ClassVar, Final, Protocol, TypeVar

import typing_extensions
from torch import device, dtype
from typing_extensions import TypeAlias, TypeGuard


class DataClass(Protocol):
    """Represents a data class object."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


def is_dataclass_instance(obj: Any) -> TypeGuard[DataClass]:
    """Return ``True`` if ``obj`` is of type :class:`DataClass`."""
    return is_dataclass(obj) and not isinstance(obj, type)


F = TypeVar("F", bound=Callable[..., Any])


override = typing_extensions.override  # compat


Device: TypeAlias = device

DataType: TypeAlias = dtype

CPU: Final = Device("cpu")

META: Final = Device("meta")
