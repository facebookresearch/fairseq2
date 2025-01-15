# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Set


class DatasetError(Exception):
    pass


class DataReadError(Exception):
    pass


class SplitNotFoundError(LookupError):
    name: str
    split: str
    available_splits: Set[str]

    def __init__(self, name: str, split: str, available_splits: Set[str]) -> None:
        s = ", ".join(sorted(available_splits))

        super().__init__(
            f"`split` must be one of the following splits, but is '{split}' instead: {s}"
        )

        self.name = name
        self.split = split
        self.available_splits = available_splits
