# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["String", "StringLike"]

from typing import TYPE_CHECKING, Optional, Union, final

from typing_extensions import TypeAlias

from fairseq2 import DOC_MODE


@final
class String:
    def __init__(self, s: Optional[str] = None) -> None:
        pass

    def __len__(self) -> int:
        pass

    def __eq__(self, other: object) -> bool:
        pass

    def __ne__(self, other: object) -> bool:
        pass

    def __hash__(self) -> int:
        pass

    def lstrip(self) -> "String":
        pass

    def rstrip(self) -> "String":
        pass

    def to_py(self) -> str:
        pass


StringLike: TypeAlias = Union[str, String]


if not TYPE_CHECKING and not DOC_MODE:
    from fairseq2._C.data.string import String  # noqa: F811

    String.__module__ = __name__
