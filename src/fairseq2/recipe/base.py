# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cache, cached_property
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
from fairseq2.recipe.internal.config import _get_config_section, _RecipeConfigHolder
from fairseq2.recipe.internal.dataset import _DatasetHolder
from fairseq2.recipe.internal.evaluator import _EvaluatorFactory
from fairseq2.recipe.internal.generator import _GeneratorFactory
from fairseq2.recipe.internal.model import _ModelHolder
from fairseq2.recipe.internal.reference_model import _ReferenceModelBootstrapper
from fairseq2.recipe.internal.tokenizer import _TokenizerHolder
from fairseq2.recipe.internal.trainer import _TrainerFactory, _ValidatorFactory
from fairseq2.recipe.model import RecipeModel, _StandardRecipeModel
from fairseq2.recipe.tokenizer import RecipeTokenizer
from fairseq2.recipe.trainer import Trainer, TrainUnit
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver

BatchT = TypeVar("BatchT", bound=SupportsDeviceTransfer)


@final
class RecipeContext:
    def __init__(self, resolver: DependencyResolver) -> None:
        self._resolver = resolver

    @cached_property
    def config(self) -> RecipeConfig:
        config_holder = self._resolver.resolve(_RecipeConfigHolder)

        return RecipeConfig(config_holder.config)

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

    @cached_property
    def model(self) -> RecipeModel:
        model_holder = self._resolver.resolve(_ModelHolder)

        return _StandardRecipeModel(model_holder)

    @cache
    def get_reference_model(self, section_name: str) -> RecipeModel:
        model_holder = self._resolver.resolve(_ModelHolder, key=section_name)

        return _StandardRecipeModel(model_holder)

    @property
    def default_dataset(self) -> RecipeDataset:
        return self.get_dataset("dataset")

    @cache
    def get_dataset(self, section_name: str) -> RecipeDataset:
        dataset_holder = self._resolver.resolve(_DatasetHolder, key=section_name)

        return RecipeDataset(
            dataset_holder.dataset, dataset_holder.config, dataset_holder.family
        )

    @property
    def default_tokenizer(self) -> RecipeTokenizer:
        return self.get_tokenizer("tokenizer")

    @cache
    def get_tokenizer(self, section_name: str) -> RecipeTokenizer:
        tokenizer_holder = self._resolver.resolve(_TokenizerHolder, key=section_name)

        return RecipeTokenizer(
            tokenizer_holder.tokenizer, tokenizer_holder.config, tokenizer_holder.family
        )

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

        model_bootstrapper = self._resolver.resolve(_ReferenceModelBootstrapper)

        model_holder = model_bootstrapper.bootstrap(section_name, section)

        return _StandardRecipeModel(model_holder)

    def create_trainer(
        self,
        unit: TrainUnit[BatchT],
        data_reader: DataReader[BatchT],
        valid_units: Sequence[EvalUnit[BatchT]] | None = None,
        valid_data_readers: Sequence[DataReader[BatchT]] | None = None,
    ) -> Trainer:
        validator_factory = self._resolver.resolve(_ValidatorFactory)

        validator = validator_factory.create(valid_units, valid_data_readers)

        factory = self._resolver.resolve(_TrainerFactory)

        return factory.create(unit, data_reader, validator)

    def create_evaluator(
        self,
        units: Sequence[EvalUnit[BatchT]],
        data_readers: Sequence[DataReader[BatchT]],
    ) -> Evaluator:
        factory = self._resolver.resolve(_EvaluatorFactory)

        return factory.create(units, data_readers)

    def create_generator(
        self, unit: GeneratorUnit[BatchT], data_reader: DataReader[BatchT]
    ) -> Generator:
        factory = self._resolver.resolve(_GeneratorFactory)

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
