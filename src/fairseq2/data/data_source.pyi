# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from enum import Enum
from typing import Any, Iterator, List, Optional

class Whence(Enum):
    FIRST = 0
    CURRENT = 1
    LAST = 2

class DataSource:
    @staticmethod
    def list_files(paths: List[str], pattern: Optional[str] = None) -> DataSource: ...
    def __iter__(self) -> Iterator[Any]: ...
    def reset(self) -> None: ...
    def seek(self, offset: int, whence: Whence = Whence.FIRST) -> None: ...
