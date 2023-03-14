# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Optional, Union, final

from typing_extensions import TypeAlias

from fairseq2 import DOC_MODE


@final
class CString:
    """Represents an immutable UTF-8 string that supports zero-copy marshalling
    between Python and native code."""

    def __init__(self, s: Optional[str] = None) -> None:
        """
        :param s:
            The Python string to copy. If ``None``, constructs an empty string.
        """

    def __len__(self) -> int:
        return 0

    def __eq__(self, other: object) -> bool:
        return False

    def __ne__(self, other: object) -> bool:
        return False

    def __hash__(self) -> int:
        return 0

    def lstrip(self) -> "CString":
        """Return a copy of this string with no whitespace at the beginning."""
        return self

    def rstrip(self) -> "CString":
        """Return a copy of this string with no whitespace at the end."""
        return self

    def to_py(self) -> str:
        """Return a copy of this string in Python."""
        return ""


StringLike: TypeAlias = Union[str, CString]


if not TYPE_CHECKING and not DOC_MODE:
    from fairseq2.C.data.string import CString  # noqa: F811

    CString.__module__ = __name__
