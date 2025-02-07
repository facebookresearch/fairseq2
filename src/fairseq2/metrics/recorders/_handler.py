# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from fairseq2.metrics.recorders._recorder import MetricRecorder


class MetricRecorderHandler(ABC):
    @abstractmethod
    def create(self, output_dir: Path, config: object) -> MetricRecorder: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...


class UnknownMetricRecorderError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known metric recorder.")

        self.name = name
