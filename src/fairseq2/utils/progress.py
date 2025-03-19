# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, final

from typing_extensions import Self, override


class ProgressReporter(ABC):
    @abstractmethod
    def create_task(
        self, name: str, total: int | None, completed: int = 0
    ) -> ProgressTask: ...

    @abstractmethod
    def __enter__(self) -> Self: ...

    @abstractmethod
    def __exit__(self, *ex: Any) -> None: ...


class ProgressTask(ABC):
    @abstractmethod
    def step(self, value: int) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *ex: Any) -> None:
        self.close()


@final
class NoopProgressReporter(ProgressReporter):
    @override
    def create_task(
        self, name: str, total: int | None, completed: int = 0
    ) -> ProgressTask:
        return NoopProgressTask()

    @override
    def __enter__(self) -> Self:
        return self

    @override
    def __exit__(self, *ex: Any) -> None:
        pass


@final
class NoopProgressTask(ProgressTask):
    @override
    def step(self, value: int) -> None:
        pass

    @override
    def close(self) -> None:
        pass
