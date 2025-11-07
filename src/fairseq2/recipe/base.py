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

from torch.nn import Module

from fairseq2.assets import AssetStore
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.datasets import DataReader
from fairseq2.device import Device, SupportsDeviceTransfer
from fairseq2.error import InvalidOperationError
from fairseq2.file_system import FileSystem
from fairseq2.gang import Gangs
from fairseq2.generation import Seq2SeqGenerator, SequenceGenerator
from fairseq2.metrics.recorders import MetricRecorder
from fairseq2.recipe.config import RecipeConfig, ReferenceModelSection
from fairseq2.recipe.dataset import RecipeDataset
from fairseq2.recipe.error import (
    DatasetTypeNotValidError,
    ModelTypeNotValidError,
    TokenizerTypeNotValidError,
)
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
from fairseq2.recipe.task import Task
from fairseq2.recipe.tokenizer import RecipeTokenizer
from fairseq2.recipe.trainer import Trainer, TrainUnit
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.warn import _warn_deprecated

BatchT = TypeVar("BatchT", bound=SupportsDeviceTransfer)

ConfigT = TypeVar("ConfigT")

ModelT = TypeVar("ModelT", bound=Module)

DatasetT = TypeVar("DatasetT")

TokenizerT = TypeVar("TokenizerT", bound=Tokenizer)


@final
class RecipeContext:
    def __init__(self, resolver: DependencyResolver) -> None:
        self._resolver = resolver

    @property
    def resolver(self) -> DependencyResolver:
        return self._resolver

    def get_config(self) -> object:
        config_holder = self._resolver.resolve(_RecipeConfigHolder)

        return config_holder.config

    def get_config_as(self, kls: type[ConfigT]) -> ConfigT:
        config = self.get_config()
        if not isinstance(config, kls):
            raise TypeError(
                f"Recipe configuration is expected to be of type `{kls}`, but is of type `{type(config)}` instead."
            )

        return config

    @property
    def output_dir(self) -> Path:
        return self._resolver.resolve(Path)

    @property
    def progress_reporter(self) -> ProgressReporter:
        return self._resolver.resolve(ProgressReporter)

    @property
    def file_system(self) -> FileSystem:
        return self._resolver.resolve(FileSystem)

    @property
    def asset_store(self) -> AssetStore:
        return self._resolver.resolve(AssetStore)

    @property
    def device(self) -> Device:
        return self._resolver.resolve(Device)

    @property
    def gangs(self) -> Gangs:
        return self._resolver.resolve(Gangs)

    @property
    def metric_recorder(self) -> MetricRecorder:
        return self._resolver.resolve(MetricRecorder)

    def get_model(self, section_name: str = "model") -> Module:
        return self._resolver.resolve(Module, key=section_name)

    def get_model_as(self, kls: type[ModelT], section_name: str = "model") -> ModelT:
        model = self.get_model(section_name)
        if not isinstance(model, kls):
            raise ModelTypeNotValidError(type(model), kls, section_name)

        return model

    def get_data_parallel_model(self) -> Module:
        return self._resolver.resolve(Module)

    def bootstrap_reference_model(self, section_name: str) -> Module:
        if section_name == "model":
            raise InvalidOperationError("`section_name` must not be 'model'.")

        section = _get_config_section(
            self._resolver, section_name, ReferenceModelSection
        )

        model_bootstrapper = self._resolver.resolve(_ReferenceModelBootstrapper)

        model_holder = model_bootstrapper.bootstrap(section_name, section)

        return model_holder.model

    def bootstrap_reference_model_as(
        self, kls: type[ModelT], section_name: str
    ) -> ModelT:
        model = self.bootstrap_reference_model(section_name)
        if not isinstance(model, kls):
            raise ModelTypeNotValidError(type(model), kls, section_name)

        return model

    def get_dataset(self, section_name: str = "dataset") -> object:
        return self._resolver.resolve(object, key=section_name)

    def get_dataset_as(
        self, kls: type[DatasetT], section_name: str = "dataset"
    ) -> DatasetT:
        dataset = self.get_dataset(section_name)
        if not isinstance(dataset, kls):
            raise DatasetTypeNotValidError(type(dataset), kls, section_name)

        return dataset

    def get_tokenizer(self, section_name: str = "tokenizer") -> Tokenizer:
        return self._resolver.resolve(Tokenizer, key=section_name)

    def get_tokenizer_as(
        self, kls: type[TokenizerT], section_name: str = "tokenizer"
    ) -> TokenizerT:
        tokenizer = self.get_tokenizer(section_name)
        if not isinstance(tokenizer, kls):
            raise TokenizerTypeNotValidError(type(tokenizer), kls, section_name)

        return tokenizer

    def get_seq_generator(self) -> SequenceGenerator:
        return self._resolver.resolve(SequenceGenerator)

    def get_seq2seq_generator(self) -> Seq2SeqGenerator:
        return self._resolver.resolve(Seq2SeqGenerator)

    def create_trainer(
        self,
        unit: TrainUnit[BatchT],
        data_reader: DataReader[BatchT],
        valid_units: Sequence[EvalUnit[BatchT]] | None = None,
        valid_data_readers: Sequence[DataReader[BatchT]] | None = None,
    ) -> Trainer:
        validator_factory = self._resolver.resolve(_ValidatorFactory)

        validator = validator_factory.create(valid_units, valid_data_readers)

        trainer_factory = self._resolver.resolve(_TrainerFactory)

        return trainer_factory.create(unit, data_reader, validator)

    def create_evaluator(
        self,
        units: Sequence[EvalUnit[BatchT]],
        data_readers: Sequence[DataReader[BatchT]],
    ) -> Evaluator:
        evaluator_factory = self._resolver.resolve(_EvaluatorFactory)

        return evaluator_factory.create(units, data_readers)

    def create_generator(
        self, unit: GeneratorUnit[BatchT], data_reader: DataReader[BatchT]
    ) -> Generator:
        generator_factory = self._resolver.resolve(_GeneratorFactory)

        return generator_factory.create(unit, data_reader)

    #
    # DEPRECATED - BEGIN
    #

    @cached_property
    def config(self) -> RecipeConfig:
        """:meta private:"""
        _warn_deprecated(
            "`RecipeContext.config` is deprecated and will be removed in v0.14. Use `RecipeContext.get_config()` or `RecipeContext.get_config_as()` instead."
        )

        config_holder = self._resolver.resolve(_RecipeConfigHolder)

        return RecipeConfig(config_holder.config)

    @cached_property
    def model(self) -> RecipeModel:
        """:meta private:"""
        _warn_deprecated(
            "`RecipeContext.model` is deprecated and will be removed in v0.14. Use `RecipeContext.get_model()`, `RecipeContext.get_model_as()`, or `RecipeContext.get_data_parallel_model()` instead."
        )

        model_holder = self._resolver.resolve(_ModelHolder)

        return _StandardRecipeModel(model_holder)

    @cache
    def get_reference_model(self, section_name: str) -> RecipeModel:
        """:meta private:"""
        _warn_deprecated(
            "`RecipeContext.get_reference_model()` is deprecated and will be removed in v0.14. Use `RecipeContext.get_model()` or `RecipeContext.get_model_as()` instead."
        )

        model_holder = self._resolver.resolve(_ModelHolder, key=section_name)

        return _StandardRecipeModel(model_holder)

    def bootstrap_model(self, section_name: str) -> RecipeModel:
        _warn_deprecated(
            "`RecipeContext.bootstrap_model()` is deprecated and will be removed in v0.14. Use `RecipeContext.bootstrap_reference_model()` or `RecipeContext.bootstrap_reference_model_as()` instead."
        )

        section = _get_config_section(
            self._resolver, section_name, ReferenceModelSection
        )

        model_bootstrapper = self._resolver.resolve(_ReferenceModelBootstrapper)

        model_holder = model_bootstrapper.bootstrap(section_name, section)

        return _StandardRecipeModel(model_holder)

    @cached_property
    def default_dataset(self) -> RecipeDataset:
        """:meta private:"""
        _warn_deprecated(
            "`RecipeContext.default_dataset` is deprecated and will be removed in v0.14. Use `RecipeContext.get_dataset()` or `RecipeContext.get_dataset_as()` instead."
        )

        dataset_holder = self._resolver.resolve(_DatasetHolder, key="dataset")

        return RecipeDataset(
            dataset_holder.dataset, dataset_holder.config, dataset_holder.family
        )

    @property
    def default_tokenizer(self) -> RecipeTokenizer:
        """:meta private:"""
        _warn_deprecated(
            "`RecipeContext.default_tokenizer` is deprecated and will be removed in v0.14. Use `RecipeContext.get_tokenizer()` or `RecipeContext.get_tokenizer_as()` instead."
        )

        tokenizer_holder = self._resolver.resolve(_TokenizerHolder, key="tokenizer")

        return RecipeTokenizer(
            tokenizer_holder.tokenizer, tokenizer_holder.config, tokenizer_holder.family
        )

    @property
    def default_seq_generator(self) -> SequenceGenerator:
        """:meta private:"""
        _warn_deprecated(
            "`RecipeContext.default_seq_generator` is deprecated and will be removed in v0.14. Use `RecipeContext.get_seq_generator()` instead."
        )

        return self._resolver.resolve(SequenceGenerator)

    @property
    def default_seq2seq_generator(self) -> Seq2SeqGenerator:
        """:meta private:"""
        _warn_deprecated(
            "`RecipeContext.default_seq2seq_generator` is deprecated and will be removed in v0.14. Use `RecipeContext.get_seq2seq_generator()` instead."
        )

        return self._resolver.resolve(Seq2SeqGenerator)

    #
    # DEPRECATED - END
    #


class Recipe(ABC):
    def register(self, container: DependencyContainer) -> None:
        pass

    def setup_model(
        self, context: RecipeContext, model: Module, newly_initialized: bool
    ) -> Module:
        return model

    def setup_reference_model(
        self, context: RecipeContext, model: Module, section_name: str
    ) -> Module:
        return model

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...

    #
    # DEPRECATED - BEGIN
    #

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

    #
    # DEPRECATED - End
    #


class TrainRecipe(Recipe):
    @abstractmethod
    def create_trainer(self, context: RecipeContext) -> Task: ...

    def has_static_autograd_graph(self, context: RecipeContext) -> bool:
        return True


class EvalRecipe(Recipe):
    @abstractmethod
    def create_evaluator(self, context: RecipeContext) -> Task: ...


class GenerationRecipe(Recipe):
    @abstractmethod
    def create_generator(self, context: RecipeContext) -> Task: ...
