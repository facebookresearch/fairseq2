# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os  # noqa: F401
from typing import Any, Union

from typing_extensions import TypeAlias, TypeGuard

from fairseq2.data.cstring import CString

# A type alias as recommended in PEP 519.
PathLike: TypeAlias = Union[str, CString, "os.PathLike[str]"]


StringLike: TypeAlias = Union[str, CString]


def is_string_like(s: Any) -> TypeGuard[StringLike]:
    return isinstance(s, (str, CString))
