# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Final, final

from typing_extensions import Self, override


class Profiler(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def step(self) -> None: ...

    def __enter__(self) -> Self:
        self.start()

        return self

    def __exit__(self, *ex: Any) -> None:
        self.stop()


@final
class _NoopProfiler(Profiler):
    @override
    def start(self) -> None:
        pass

    @override
    def stop(self) -> None:
        pass

    @override
    def step(self) -> None:
        pass


NOOP_PROFILER: Final = _NoopProfiler()
