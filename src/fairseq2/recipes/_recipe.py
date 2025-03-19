# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod


class Recipe(ABC):
    @abstractmethod
    def run(self) -> None: ...

    @abstractmethod
    def request_stop(self) -> None: ...

    @property
    @abstractmethod
    def step_nr(self) -> int: ...


class RecipeStopException(Exception):
    def __init__(self) -> None:
        super().__init__("The recipe has been stopped.")
