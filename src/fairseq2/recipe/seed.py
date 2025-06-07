# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from fairseq2.recipe.config import CommonSection


@final
class SeedHolder:
    def __init__(self, section: CommonSection) -> None:
        self._value = section.seed

    def advance(self) -> int:
        value = self._value

        self._value += 1

        return value
