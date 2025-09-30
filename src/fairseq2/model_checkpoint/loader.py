# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Protocol

from torch import Tensor

from fairseq2.gang import Gangs
from fairseq2.sharder import ShardSpec


class StateDictConverter(Protocol):
    def __call__(self, state_dict: dict[str, object]) -> dict[str, object]: ...


class ModelCheckpointLoader(ABC):
    @abstractmethod
    def lazy_load(
        self,
        path: Path,
        gangs: Gangs,
        *,
        mmap: bool = False,
        restrict: bool = True,
        state_dict_converter: StateDictConverter | None = None,
        shard_specs: Mapping[str, ShardSpec] | None = None,
    ) -> Iterator[tuple[str, Tensor]]: ...

    @abstractmethod
    def supports_path(self, path: Path) -> bool: ...


class ModelCheckpointError(Exception):
    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)

        self.path = path
