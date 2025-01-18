# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Set


class UnknownDatasetError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known dataset.")

        self.name = name


class UnknownDatasetFamilyError(Exception):
    family: str
    dataset_name: str | None

    def __init__(self, family: str, dataset_name: str | None = None) -> None:
        super().__init__(f"'{family}' is not a known dataset family.")

        self.family = family
        self.dataset_name = dataset_name


class DatasetLoadError(Exception):
    name: str

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)

        self.name = name


class DataReadError(Exception):
    name: str

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)

        self.name = name


class SplitNotFoundError(DataReadError):
    split: str
    available_splits: Set[str]

    def __init__(self, name: str, split: str, available_splits: Set[str]) -> None:
        s = ", ".join(sorted(available_splits))

        super().__init__(
            name, f"`split` must be one of the following splits, but is '{split}' instead: {s}"  # fmt: skip
        )

        self.split = split
        self.available_splits = available_splits
