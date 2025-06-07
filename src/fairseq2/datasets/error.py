# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Set


class UnknownDatasetError(Exception):
    dataset_name: str

    def __init__(self, dataset_name: str) -> None:
        super().__init__(f"'{dataset_name}' is not a known dataset.")

        self.dataset_name = dataset_name


class UnknownDatasetFamilyError(Exception):
    family: str

    def __init__(self, family: str) -> None:
        super().__init__(f"'{family}' is not a know dataset family.")

        self.family = family


class DatasetError(Exception):
    dataset_name: str

    def __init__(self, dataset_name: str, message: str) -> None:
        super().__init__(message)

        self.dataset_name = dataset_name


class UnknownSplitError(ValueError):
    dataset_name: str
    split: str
    available_splits: Set[str]

    def __init__(
        self, dataset_name: str, split: str, available_splits: Set[str]
    ) -> None:
        s = ", ".join(sorted(available_splits))

        super().__init__(
            f"'{split}' is not a known split of the '{dataset_name}' dataset. The following splits are available: {s}"
        )

        self.dataset_name = dataset_name
        self.split = split
        self.available_splits = available_splits
