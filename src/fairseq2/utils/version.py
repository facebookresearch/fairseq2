# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Final

import torch
from packaging import version
from packaging.version import InvalidVersion, Version


def _get_torch_version() -> Version:
    try:
        return version.parse(torch.__version__)
    except InvalidVersion:
        return Version("0.0.0")


TORCH_VERSION: Final = _get_torch_version()


def torch_greater_or_equal(major: int, minor: int) -> bool:
    if TORCH_VERSION.major <= major - 1:
        return False

    if TORCH_VERSION.major == major and TORCH_VERSION.minor <= minor - 1:
        return False

    return True
