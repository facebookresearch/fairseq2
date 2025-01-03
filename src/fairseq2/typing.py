# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import Field, is_dataclass
from typing import Any, ClassVar, Final, Protocol, TypeAlias, TypeGuard, TypeVar

from torch import device, dtype
from typing_extensions import override as override  # noqa: F401

T = TypeVar("T")


def safe_cast(param_name: str, value: object, kls: type[T]) -> T:
    if not isinstance(value, kls):
        raise TypeError(
            f"`{param_name}` must be of type `{kls}`, but is of type `{type(value)}` instead."
        )

    return value


class DataClass(Protocol):
    """Represents a data class object."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


def is_dataclass_instance(obj: object) -> TypeGuard[DataClass]:
    """Return ``True`` if ``obj`` is of type :class:`DataClass`."""
    return is_dataclass(obj) and not isinstance(obj, type)


Device: TypeAlias = device

DataType: TypeAlias = dtype

CPU: Final = Device("cpu")

META: Final = Device("meta")
