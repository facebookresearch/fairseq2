# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, List, Optional, overload

from fairseq2n import DOC_MODE

if TYPE_CHECKING or DOC_MODE:

    class CString:
        """
        Represents an immutable UTF-8 string that supports zero-copy marshalling
        between Python and native code.
        """

        @overload
        def __init__(self) -> None:
            ...

        @overload
        def __init__(self, s: str) -> None:
            ...

        def __init__(self, s: Optional[str] = None) -> None:
            """
            :param s:
                The Python string to copy.
            """

        def __len__(self) -> int:
            ...

        def __eq__(self, other: object) -> bool:
            ...

        def __ne__(self, other: object) -> bool:
            ...

        def __hash__(self) -> int:
            ...

        def __bytes__(self) -> bytes:
            ...

        def strip(self) -> "CString":
            """Return a copy of this string with no whitespace at the beginning and end."""

        def lstrip(self) -> "CString":
            """Return a copy of this string with no whitespace at the beginning."""

        def rstrip(self) -> "CString":
            """Return a copy of this string with no whitespace at the end."""

        def split(self, sep: Optional[str] = None) -> List["CString"]:
            """Return a list of the words in string using sep as the delimiter string."""

else:
    from fairseq2n.bindings.data.string import CString as CString

    CString.__module__ = __name__
