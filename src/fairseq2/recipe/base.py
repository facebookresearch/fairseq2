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
from fairseq2.datasets import DataReader
from fairseq2.device import SupportsDeviceTransfer
from fairseq2.file_system import FileSystem
from fairseq2.gang import Gangs
from fairseq2.generation import Seq2SeqGenerator, SequenceGenerator
from fairseq2.recipe.config import RecipeConfig, ReferenceModelSection
from fairseq2.recipe.dataset import RecipeDataset
from fairseq2.recipe.evaluator import Evaluator, EvalUnit
from fairseq2.recipe.generator import Generator, GeneratorUnit
from fairseq2.recipe.internal.config import _get_config_section
from fairseq2.recipe.internal.eval_model import _EvalModelBootstrapper
from fairseq2.recipe.internal.evaluator import _RecipeEvaluatorFactory
from fairseq2.recipe.internal.generator import _RecipeGeneratorFactory
from fairseq2.recipe.internal.trainer import (
    _RecipeTrainerFactory,
    _RecipeValidatorFactory,
)
from fairseq2.recipe.model import RecipeModel
from fairseq2.recipe.tokenizer import RecipeTokenizer
from fairseq2.recipe.trainer import Trainer, TrainUnit
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver

BatchT = TypeVar("BatchT", bound=SupportsDeviceTransfer)


@final
class RecipeContext:
    def __init__(self, resolver: DependencyResolver) -> None:
        self._resolver = resolver

    @property
    def config(self) -> RecipeConfig:
        return self._resolver.resolve(RecipeConfig)

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
    def model(self) -> RecipeModel:
        return self._resolver.resolve(RecipeModel)

    def get_reference_model(self, section_name: str) -> RecipeModel:
        return self._resolver.resolve(RecipeModel, key=section_name)

    @property
    def default_dataset(self) -> RecipeDataset:
        return self.get_dataset("dataset")

    def get_dataset(self, section_name: str) -> RecipeDataset:
        return self._resolver.resolve(RecipeDataset, key=section_name)

    @property
    def default_tokenizer(self) -> RecipeTokenizer:
        return self.get_tokenizer("tokenizer")

    def get_tokenizer(self, section_name: str) -> RecipeTokenizer:
        return self._resolver.resolve(RecipeTokenizer, key=section_name)

    @property
    def default_seq_generator(self) -> SequenceGenerator:
        return self._resolver.resolve(SequenceGenerator)

    @property
    def default_seq2seq_generator(self) -> Seq2SeqGenerator:
        return self._resolver.resolve(Seq2SeqGenerator)

    @property
    def resolver(self) -> DependencyResolver:
        return self._resolver

    def bootstrap_model(self, section_name: str) -> RecipeModel:
        section = _get_config_section(
            self._resolver, section_name, ReferenceModelSection
        )

        model_bootstrapper = self._resolver.resolve(_EvalModelBootstrapper)

        return model_bootstrapper.bootstrap(section_name, section)

    def create_trainer(
        self,
        unit: TrainUnit[BatchT],
        data_reader: DataReader[BatchT],
        valid_units: Sequence[EvalUnit[BatchT]] | None = None,
        valid_data_readers: Sequence[DataReader[BatchT]] | None = None,
    ) -> Trainer:
        validator_factory = self._resolver.resolve(_RecipeValidatorFactory)

        validator = validator_factory.create(valid_units, valid_data_readers)

        factory = self._resolver.resolve(_RecipeTrainerFactory)

        return factory.create(unit, data_reader, validator)

    def create_evaluator(
        self,
        units: Sequence[EvalUnit[BatchT]],
        data_readers: Sequence[DataReader[BatchT]],
    ) -> Evaluator:
        factory = self._resolver.resolve(_RecipeEvaluatorFactory)

        return factory.create(units, data_readers)

    def create_generator(
        self, unit: GeneratorUnit[BatchT], data_reader: DataReader[BatchT]
    ) -> Generator:
        factory = self._resolver.resolve(_RecipeGeneratorFactory)

        return factory.create(unit, data_reader)


class Recipe(ABC):
    def register(self, container: DependencyContainer) -> None:
        pass

    def prepare_model(self, context: RecipeContext, model: RecipeModel) -> RecipeModel:
        return model

    def prepare_reference_model(
        self,
        context: RecipeContext,
        model: RecipeModel,
        section_name: str,
        section: ReferenceModelSection,
    ) -> RecipeModel:
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
