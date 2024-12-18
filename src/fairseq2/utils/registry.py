# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, final

from fairseq2.error import AlreadyExistsError

T_co = TypeVar("T_co", covariant=True)


class Provider(ABC, Generic[T_co]):
    @abstractmethod
    def get(self, key: str) -> T_co:
        ...


T = TypeVar("T")


@final
class Registry(Provider[T]):
    _entries: dict[str, T]

    def __init__(self) -> None:
        self._entries = {}

    def get(self, key: str) -> T:
        try:
            return self._entries[key]
        except KeyError:
            raise LookupError(f"The registry does not contain a '{key}' key.") from None

    def register(self, key: str, value: T) -> None:
        if key in self._entries:
            raise AlreadyExistsError(f"The registry already contains a '{key}' key.")

        self._entries[key] = value
