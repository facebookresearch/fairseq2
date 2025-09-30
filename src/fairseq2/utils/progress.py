# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Final, final

from typing_extensions import Self, override

from fairseq2.runtime.closable import Closable


class ProgressReporter(ABC):
    @abstractmethod
    def create_task(
        self, name: str, total: int | None, completed: int = 0, *, start: bool = True
    ) -> ProgressTask: ...

    @abstractmethod
    def __enter__(self) -> Self: ...

    @abstractmethod
    def __exit__(self, *ex: Any) -> None: ...


class ProgressTask(Closable):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def step(self, value: int = 1) -> None: ...

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *ex: Any) -> None:
        self.close()


@final
class _NoopProgressReporter(ProgressReporter):
    @override
    def create_task(
        self, name: str, total: int | None, completed: int = 0, *, start: bool = True
    ) -> ProgressTask:
        return _NoopProgressTask()

    @override
    def __enter__(self) -> Self:
        return self

    @override
    def __exit__(self, *ex: Any) -> None:
        pass


NOOP_PROGRESS_REPORTER: Final = _NoopProgressReporter()


@final
class _NoopProgressTask(ProgressTask):
    @override
    def start(self) -> None:
        pass

    @override
    def step(self, value: int = 1) -> None:
        pass

    @override
    def close(self) -> None:
        pass
