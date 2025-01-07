# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations


class AssetError(Exception):
    name: str

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)

        self.name = name


class AssetCardError(AssetError):
    pass


class AssetCardNotFoundError(AssetCardError):
    pass


class AssetCardFieldNotFoundError(AssetCardError):
    pass
