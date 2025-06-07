# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar, final

from fairseq2.assets import AssetStore
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.datasets import DataReader
from fairseq2.device import SupportsDeviceTransfer
from fairseq2.evaluator import Evaluator, EvalUnit
from fairseq2.file_system import FileSystem
from fairseq2.gang import Gangs
from fairseq2.generation import Seq2SeqGenerator, SequenceGenerator
from fairseq2.generator import Generator, GeneratorUnit
from fairseq2.model import Model
from fairseq2.recipe.config import ReferenceModelSection
from fairseq2.recipe.evaluator import _create_evaluator
from fairseq2.recipe.generator import _create_generator
from fairseq2.recipe.seed import SeedHolder
from fairseq2.recipe.trainer import _create_trainer
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver
from fairseq2.trainer import Trainer, TrainUnit

BatchT = TypeVar("BatchT", bound=SupportsDeviceTransfer)

T = TypeVar("T")


@final
class RecipeContext:
    def __init__(self, resolver: DependencyResolver) -> None:
        self._resolver = resolver

    def config_as(self, kls: type[T]) -> T:
        config = self._resolver.resolve(object, key="config")

        if not isinstance(config, kls):
            raise TypeError(
                f"Recipe configuration is expected to be of type `{kls}`, but is of type `{type(config)}` instead."
            )

        return config

    @property
    def output_dir(self) -> Path:
        return self._resolver.resolve(Path)

    @property
    def file_system(self) -> FileSystem:
        return self._resolver.resolve(FileSystem)

    @property
    def asset_store(self) -> AssetStore:
        return self._resolver.resolve(AssetStore)

    @property
    def gangs(self) -> Gangs:
        return self._resolver.resolve(Gangs)

    @property
    def model(self) -> Model:
        return self._resolver.resolve(Model)

    def dataset_as(self, kls: type[T]) -> T:
        dataset = self._resolver.resolve(object, key="dataset")

        if not isinstance(dataset, kls):
            raise TypeError(
                f"Dataset is expected to be of type `{kls}`, but is of type `{type(dataset)}` instead."
            )

        return dataset

    @property
    def tokenizer(self) -> Tokenizer:
        return self._resolver.resolve(Tokenizer)

    @property
    def source_tokenizer(self) -> Tokenizer:
        return self._resolver.resolve(Tokenizer, key="source")

    @property
    def seq_generator(self) -> SequenceGenerator:
        return self._resolver.resolve(SequenceGenerator)

    @property
    def seq2seq_generator(self) -> Seq2SeqGenerator:
        return self._resolver.resolve(Seq2SeqGenerator)

    @property
    def resolver(self) -> DependencyResolver:
        return self._resolver

    def next_seed(self) -> int:
        return self._resolver.resolve(SeedHolder).advance()

    def create_trainer(
        self,
        unit: TrainUnit[BatchT],
        data_reader: DataReader[BatchT],
        valid_units: Sequence[EvalUnit[BatchT]] | None = None,
        valid_data_readers: Sequence[DataReader[BatchT]] | None = None,
    ) -> Trainer:
        return _create_trainer(
            self._resolver, unit, data_reader, valid_units, valid_data_readers
        )

    def create_evaluator(
        self,
        units: Sequence[EvalUnit[BatchT]],
        data_readers: Sequence[DataReader[BatchT]],
    ) -> Evaluator:
        return _create_evaluator(self._resolver, units, data_readers)

    def create_generator(
        self, unit: GeneratorUnit[BatchT], data_reader: DataReader[BatchT]
    ) -> Generator:
        return _create_generator(self._resolver, unit, data_reader)


class Recipe(ABC):
    def register(self, container: DependencyContainer) -> None:
        pass

    def prepare_model(self, context: RecipeContext, model: Model) -> Model:
        return model

    def prepare_eval_model(
        self,
        context: RecipeContext,
        model: Model,
        section_name: str,
        section: ReferenceModelSection,
    ) -> Model:
        return model

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...


class TrainRecipe(Recipe):
    @abstractmethod
    def create_trainer(self, context: RecipeContext) -> Trainer: ...

    def has_static_autograd_graph(self, context: RecipeContext) -> bool:
        return True


class EvalRecipe(Recipe):
    @abstractmethod
    def create_evaluator(self, context: RecipeContext) -> Evaluator: ...


class GenerationRecipe(Recipe):
    @abstractmethod
    def create_generator(self, context: RecipeContext) -> Generator: ...
