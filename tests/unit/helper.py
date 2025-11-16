# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator
from typing import final, overload

from typing_extensions import override

from fairseq2.utils.env import Environment


@final
class FooEnvironment(Environment):
    def __init__(self, data: dict[str, str] = {}) -> None:
        self._data = data

    @override
    def get(self, name: str) -> str:
        return self._data[name]

    @overload
    def maybe_get(self, name: str) -> str | None: ...

    @overload
    def maybe_get(self, name: str, default: str) -> str: ...

    @override
    def maybe_get(self, name: str, default: str | None = None) -> str | None:
        return self._data.get(name, default)

    @override
    def set(self, name: str, value: str) -> None:
        self._data[name] = value

    @override
    def has(self, name: str) -> bool:
        return name in self._data

    @override
    def to_dict(self) -> dict[str, str]:
        return dict(self._data)

    @override
    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(self._data.items())
