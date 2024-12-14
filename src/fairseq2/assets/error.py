# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TypeAlias


class AssetError(Exception):
    pass


class AssetCardError(AssetError):
    pass


class AssetCardNotFoundError(AssetCardError):
    pass


class AssetCardFieldNotFoundError(AssetCardError):
    pass


AssetNotFoundError: TypeAlias = AssetCardNotFoundError  # compat
