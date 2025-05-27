# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import MutableMapping
from contextlib import AbstractContextManager
from dataclasses import Field, is_dataclass
from typing import Any, ClassVar, Protocol, TypeAlias, TypeGuard

from typing_extensions import Self

ContextManager: TypeAlias = AbstractContextManager[None]


class DataClass(Protocol):
    """Represents a data class object."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


def is_dataclass_instance(obj: object) -> TypeGuard[DataClass]:
    """Return ``True`` if ``obj`` is of type :class:`DataClass`."""
    return is_dataclass(obj) and not isinstance(obj, type)


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
"""A sentinel signifying no value."""


def get_name_or_self(obj: object) -> object:
    return getattr(obj, "__name__", obj)


class Closable(Protocol):
    def close(self) -> None: ...


class Compilable(Protocol):
    def compile(self, *args: Any, **kwargs: Any) -> object: ...
