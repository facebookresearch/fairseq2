# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from typing import Any, final

from typing_extensions import Self, override

from fairseq2.logging import log


class GarbageCollector(ABC):
    @abstractmethod
    def enable(self, value: bool = True) -> None: ...

    @abstractmethod
    def step(self) -> None: ...

    def __enter__(self) -> Self:
        self.enable()

        return self

    def __exit__(self, *ex: Any) -> None:
        self.enable(False)


@final
class NoopGarbageCollector(GarbageCollector):
    @override
    def enable(self, value: bool = True) -> None:
        pass

    @override
    def step(self) -> None:
        pass


@final
class CPythonGarbageCollector(GarbageCollector):
    _step: int
    _collect_every_n_step: int

    def __init__(self, collect_every_n_step: int) -> None:
        if collect_every_n_step < 1:
            raise ValueError(
                "`collect_every_n_step` must be greater than or equal to 1."
            )

        self._step = 0
        self._collect_every_n_step = collect_every_n_step

    @override
    def enable(self, value: bool = True) -> None:
        if value:
            gc.disable()

            gc.collect()
        else:
            gc.enable()

    @override
    def step(self) -> None:
        self._step += 1

        if self._step == self._collect_every_n_step:
            log.info("Running garbage collection for the oldest two generations.")  # fmt: skip

            gc.collect(generation=1)

            self._step = 0
