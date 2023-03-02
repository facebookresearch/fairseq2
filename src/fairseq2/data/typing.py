# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os  # noqa: F401
from typing import Union

from typing_extensions import TypeAlias

from fairseq2.data.string import String

# A type alias as recommended in PEP 519.
PathLike: TypeAlias = Union[str, String, "os.PathLike[str]"]
