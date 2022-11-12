# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor
from typing_extensions import TypeAlias

# fmt: off

class IString:
    def __init__(self, s: str | None = None) -> None:
        """
        :param s:
            The string to copy.
        """

    def __len__(self) -> int:
        ...

    def __eq__(self, other: object) -> bool | NotImplemented:  # type: ignore[valid-type]
        ...

    def __ne__(self, other: object) -> bool | NotImplemented:  # type: ignore[valid-type]
        ...

    def __hash__(self) -> int:
        ...

    def to_py(self) -> str:
        """Converts to ``str``.

        :returns:
            A ``str`` representation of this string.
        """


IVariant: TypeAlias = None | bool | int | float | str | IString | Tensor
