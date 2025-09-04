# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar, final

T_co = TypeVar("T_co", covariant=True)


@final
class Lazy(Generic[T_co]):
    _value: T_co

    def __init__(self, factory: Callable[[], T_co]):
        self._factory = factory

    def get(self) -> T_co:
        if not hasattr(self, "_value"):
            self._value = self._factory()

            del self._factory

        return self._value
