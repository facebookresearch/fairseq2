# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os  # noqa: F401
from typing import Any, Union

from typing_extensions import TypeAlias, TypeGuard

from fairseq2.data.cstring import CString

# A type alias for pathnames as recommended in PEP 519.
PathLike: TypeAlias = Union[str, CString, "os.PathLike[str]"]


# A convenience type alias for strings since most of our data APIs accept both
# `str` and `CString`.
StringLike: TypeAlias = Union[str, CString]


def is_string_like(s: Any) -> TypeGuard[StringLike]:
    """Return ``True`` if ``s`` is of type ``str`` or :class:`CString`."""
    return isinstance(s, (str, CString))
