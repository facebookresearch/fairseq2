# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from typing_extensions import override

from fairseq2.profilers.profiler import Profiler


@final
class CompositeProfiler(Profiler):
    def __init__(self, profilers: Sequence[Profiler]) -> None:
        self._profilers = profilers

    @override
    def start(self) -> None:
        for profiler in self._profilers:
            profiler.start()

    @override
    def stop(self) -> None:
        for profiler in self._profilers:
            profiler.stop()

    @override
    def step(self) -> None:
        for profiler in self._profilers:
            profiler.step()
