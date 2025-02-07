# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from fairseq2.gang import Gangs
from fairseq2.profilers._profiler import Profiler


class ProfilerHandler(ABC):
    @abstractmethod
    def create(self, config: object, gangs: Gangs, output_dir: Path) -> Profiler: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...
