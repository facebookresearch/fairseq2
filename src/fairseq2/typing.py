# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import Field, is_dataclass
from typing import Any, ClassVar, Protocol, TypeAlias, TypeGuard, runtime_checkable

ContextManager: TypeAlias = AbstractContextManager


@runtime_checkable
class Stateful(Protocol):
    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state_dict: dict[str, object]) -> None: ...


class DataClass(Protocol):
    """Represents a data class object."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


def is_dataclass_instance(obj: object) -> TypeGuard[DataClass]:
    """Return ``True`` if ``obj`` is of type :class:`DataClass`."""
    return is_dataclass(obj) and not isinstance(obj, type)
