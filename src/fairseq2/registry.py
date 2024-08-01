# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Holds objects of type ``T``."""

    _objects: Dict[str, T]

    def __init__(self) -> None:
        self._objects = {}

    def get(self, name: str) -> T:
        """Return the object registered with ``name``."""
        try:
            return self._objects[name]
        except KeyError:
            raise ValueError(
                f"`name` must be a registered name, but is '{name}' instead."
            ) from None

    def register(self, name: str, obj: T) -> None:
        """Register ``obj`` with ``name``."""
        if name in self._objects:
            raise ValueError(
                f"`name` must be a unique name, but '{name}' is already registered."
            )

        self._objects[name] = obj
