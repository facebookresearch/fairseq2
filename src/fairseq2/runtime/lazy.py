# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar, cast, final

T_co = TypeVar("T_co", covariant=True)


@final
class Lazy(Generic[T_co]):
    _value: T_co | Callable[[], T_co]
    _constructed: bool

    def __init__(self, factory: Callable[[], T_co]):
        self._value = factory
        self._constructed = False

    def retrieve(self) -> T_co:
        if not self._constructed:
            self._value = cast(Callable[[], T_co], self._value)()

            self._constructed = True

        return cast(T_co, self._value)
