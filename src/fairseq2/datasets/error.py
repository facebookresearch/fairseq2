# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Set


class SplitNotKnownError(Exception):
    def __init__(
        self, dataset_name: str, split: str, available_splits: Set[str]
    ) -> None:
        s = ", ".join(sorted(available_splits))

        super().__init__(
            f"{split} is not a known split of the {dataset_name} dataset. Available splits are {s}."
        )

        self.dataset_name = dataset_name
        self.split = split
        self.available_splits = available_splits
