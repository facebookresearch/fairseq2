# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

from torch import Tensor
from typing_extensions import TypeAlias

class IString:
    def __init__(self, s: Optional[str] = None) -> None: ...
    def __len__(self) -> int: ...
    def to_py(self) -> str: ...

IVariant: TypeAlias = Union[None, bool, int, float, str, IString, Tensor]
