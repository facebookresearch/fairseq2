# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import final, overload

# fmt: off

@final
class String:
    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, s: str) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __eq__(self, other: object) -> bool | NotImplemented:  # type: ignore[valid-type]
        ...

    def __ne__(self, other: object) -> bool | NotImplemented:  # type: ignore[valid-type]
        ...

    def __hash__(self) -> int:
        ...

    def lstrip(self) -> String:
        ...

    def rstrip(self) -> String:
        ...

    def to_py(self) -> str:
        ...
