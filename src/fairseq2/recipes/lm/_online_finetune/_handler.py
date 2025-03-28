# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod

from torch.nn import Module

from fairseq2.models.sequence import SequenceBatch
from fairseq2.gang import Gangs
from fairseq2.recipes.trainer import TrainUnit


class OnlineFinetuneUnitHandler(ABC):
    @abstractmethod
    def create(
        self, model: Module, gangs: Gangs, recipe_config: object, vllm_actors: object
    ) -> TrainUnit[SequenceBatch]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...


class UnknownOnlineFinetuneUnitError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known online optimization unit.")

        self.name = name
