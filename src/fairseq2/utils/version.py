# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Final

import torch
from packaging import version
from packaging.version import InvalidVersion, Version


def _get_torch_version() -> Version:
    try:
        return version.parse(torch.__version__)
    except InvalidVersion:
        return Version("0.0.0")


TORCH_VERSION: Final[Version] = _get_torch_version()


def is_pt2_or_greater() -> bool:
    """Return ``True`` if the version of PyTorch is 2.0 or greater."""
    return TORCH_VERSION.major >= 2
