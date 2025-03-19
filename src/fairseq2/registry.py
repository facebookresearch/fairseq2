# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable
from typing import Generic, TypeVar, final

from typing_extensions import override

from fairseq2.error import AlreadyExistsError

T_co = TypeVar("T_co", covariant=True)


class Provider(ABC, Generic[T_co]):
    @abstractmethod
    def get(self, key: Hashable) -> T_co: ...

    @abstractmethod
    def get_all(self) -> Iterable[tuple[Hashable, T_co]]: ...

    @abstractmethod
    def has(self, key: Hashable) -> bool: ...

    @property
    @abstractmethod
    def kls(self) -> type[T_co]: ...


T = TypeVar("T")


@final
class Registry(Provider[T]):
    _entries: dict[Hashable, T]
    _kls: type[T]

    def __init__(self, kls: type[T]) -> None:
        self._entries = {}
        self._kls = kls

    @override
    def get(self, key: Hashable) -> T:
        try:
            return self._entries[key]
        except KeyError:
            raise LookupError(f"The registry does not contain a '{key}' key.") from None

    @override
    def get_all(self) -> Iterable[tuple[Hashable, T]]:
        return self._entries.items()

    @override
    def has(self, key: Hashable) -> bool:
        return key in self._entries

    def register(self, key: Hashable, value: T) -> None:
        if key in self._entries:
            raise AlreadyExistsError(f"The registry already contains a '{key}' key.")

        self._entries[key] = value

    @property
    @override
    def kls(self) -> type[T]:
        return self._kls
