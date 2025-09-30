# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator
from typing import Generic, TypeVar

T_co = TypeVar("T_co", covariant=True)


class Lookup(ABC, Generic[T_co]):
    @abstractmethod
    def maybe_get(self, key: Hashable) -> T_co | None: ...

    @abstractmethod
    def iter_keys(self) -> Iterator[Hashable]: ...

    @property
    @abstractmethod
    def kls(self) -> type[T_co]: ...
