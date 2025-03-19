# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations


class UnknownLRSchedulerError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known learning rate scheduler.")

        self.name = name


class UnspecifiedNumberOfStepsError(ValueError):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(
            f"`num_steps` must be specified for the '{name}' learning rate scheduler."
        )

        self.name = name
