# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import Field, is_dataclass
from typing import Any, ClassVar, Final, Protocol, TypeAlias, TypeGuard

from torch import device, dtype
from typing_extensions import Self
from typing_extensions import override as override  # noqa: F401


class DataClass(Protocol):
    """Represents a data class object."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


class _EmptyType:
    def __reduce__(self) -> str:
        return "EMPTY"

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: MutableMapping[Any, Any]) -> Self:
        return self

    def __repr__(self) -> str:
        return "<empty>"


EMPTY = _EmptyType()
"""A sentinel signifying no value for a dataclass field."""


def is_dataclass_instance(obj: object) -> TypeGuard[DataClass]:
    """Return ``True`` if ``obj`` is of type :class:`DataClass`."""
    return is_dataclass(obj) and not isinstance(obj, type)


Device: TypeAlias = device

DataType: TypeAlias = dtype

CPU: Final = Device("cpu")

META: Final = Device("meta")
