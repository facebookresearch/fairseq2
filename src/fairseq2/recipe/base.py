# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeVar, cast, final

from fairseq2.data.tokenizers import Tokenizer
from fairseq2.datasets import DataReader
from fairseq2.dependency import DependencyContainer, DependencyResolver
from fairseq2.device import SupportsDeviceTransfer
from fairseq2.gang import Gangs
from fairseq2.recipe.base_model import load_base_model
from fairseq2.recipe.config import get_config
from fairseq2.recipe.evaluator import Evaluator, EvalUnit, create_evaluator
from fairseq2.recipe.generator import Generator, GeneratorUnit, create_generator
from fairseq2.recipe.model import Model
from fairseq2.recipe.reference_model import (
    load_base_eval_model,
    load_base_generator_model,
)
from fairseq2.recipe.trainer import Trainer, TrainUnit, create_trainer
from fairseq2.utils.rng import SeedHolder

BatchT = TypeVar("BatchT", bound=SupportsDeviceTransfer)

ConfigT = TypeVar("ConfigT")


@final
class RecipeContext:
    _resolver: DependencyResolver

    def __init__(self, resolver: DependencyResolver) -> None:
        self._resolver = resolver

    def create_trainer(
        self,
        unit: TrainUnit[BatchT],
        data_reader: DataReader[BatchT],
        valid_units: Sequence[EvalUnit[BatchT]] | None = None,
        valid_data_readers: Sequence[DataReader[BatchT]] | None = None,
    ) -> Trainer:
        return create_trainer(
            self._resolver, unit, data_reader, valid_units, valid_data_readers
        )

    def create_evaluator(
        self,
        units: Sequence[EvalUnit[BatchT]],
        data_readers: Sequence[DataReader[BatchT]],
    ) -> Evaluator:
        return create_evaluator(self._resolver, units, data_readers)

    def create_generator(
        self, unit: GeneratorUnit[BatchT], data_reader: DataReader[BatchT]
    ) -> Generator:
        return create_generator(self._resolver, unit, data_reader)

    def next_seed(self) -> int:
        return self._resolver.resolve(SeedHolder).advance()

    @property
    def gangs(self) -> Gangs:
        return self._resolver.resolve(Gangs)

    @property
    def model(self) -> Model:
        return self._resolver.resolve(Model)

    @property
    def tokenizer(self) -> Tokenizer:
        return self._resolver.resolve(Tokenizer)

    @property
    def config(self) -> object:
        return get_config(self._resolver)

    def config_as(self, kls: type[ConfigT]) -> ConfigT:
        return cast(ConfigT, self.config)

    @property
    def resolver(self) -> DependencyResolver:
        return self._resolver


class TrainRecipe(ABC):
    def register(self, container: DependencyContainer) -> None:
        pass

    def load_base_model(self, context: RecipeContext) -> Model:
        return load_base_model(context.resolver)

    def has_static_autograd_graph(self, context: RecipeContext) -> bool:
        return True

    @abstractmethod
    def load_trainer(self, context: RecipeContext) -> Trainer: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...


class EvalRecipe(ABC):
    def register(self, container: DependencyContainer) -> None:
        pass

    def load_base_model(self, context: RecipeContext) -> Model:
        return load_base_eval_model(context.resolver)

    @abstractmethod
    def load_evaluator(self, context: RecipeContext) -> Evaluator: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...


class GenerationRecipe(ABC):
    def register(self, container: DependencyContainer) -> None:
        pass

    def load_base_model(self, context: RecipeContext) -> Model:
        return load_base_generator_model(context.resolver)

    @abstractmethod
    def load_generator(self, context: RecipeContext) -> Generator: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...
