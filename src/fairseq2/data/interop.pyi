# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# fmt: off

class IString:
    """Represents an immutable UTF-8 string that supports zero-copy marshalling
    between Python and native code."""

    def __init__(self, s: str | None = None) -> None:
        """
        :param s:
            The Python string to copy.
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


class IList:
    """Holds :data:`IVariant` elements that can be zero-copy marshalled between
    Python and native code."""

class IDict:
    """Holds :data:`IVariant` key/value pairs that can be zero-copy marshalled
    between Python and native code."""
