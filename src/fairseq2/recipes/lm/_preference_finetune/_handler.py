# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod

from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.gang import Gangs
from fairseq2.recipes import Model, TrainUnit


class POFinetuneUnitHandler(ABC):
    @abstractmethod
    def create(
        self, model: Model, gangs: Gangs, recipe_config: object
    ) -> TrainUnit[PreferenceBatch]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...


class UnknownPOFinetuneUnitError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known preference optimization unit.")

        self.name = name
