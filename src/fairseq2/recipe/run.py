# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import final

import torch

import fairseq2.runtime.dependency
from fairseq2.composition import _register_library
from fairseq2.error import raise_operational_system_error
from fairseq2.recipe.base import EvalRecipe, GenerationRecipe, Recipe, TrainRecipe
from fairseq2.recipe.config import RecipeConfig
from fairseq2.recipe.internal.cluster import _ClusterPreparer
from fairseq2.recipe.internal.config_preparer import _RecipeConfigPreparer
from fairseq2.recipe.internal.log import _LogHelper
from fairseq2.recipe.internal.logging import _DistributedLogConfigurer
from fairseq2.recipe.internal.output_dir import _OutputDirectoryCreator
from fairseq2.recipe.internal.task import _TaskRunner
from fairseq2.recipe.internal.torch import _TorchConfigurer
from fairseq2.recipe.task import Task
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver
from fairseq2.utils.rich import configure_rich_logging
from fairseq2.utils.structured import ValueConverter
from fairseq2.utils.warn import enable_deprecation_warnings
from fairseq2.utils.yaml import YamlDumper
from fairseq2.world_info import WorldInfo


def train(recipe: TrainRecipe, config: object, output_dir: Path) -> None:
    from fairseq2.recipe.composition import _register_train_recipe

    enable_deprecation_warnings()

    configure_rich_logging()

    container = DependencyContainer()

    with _swap_default_resolver(container):
        _register_library(container)

        _register_train_recipe(container, recipe)

        _register_run(container, recipe, config, output_dir)

        _run_recipe(container)


@torch.inference_mode()
def evaluate(recipe: EvalRecipe, config: object, output_dir: Path) -> None:
    from fairseq2.recipe.composition import _register_eval_recipe

    enable_deprecation_warnings()

    configure_rich_logging()

    container = DependencyContainer()

    with _swap_default_resolver(container):
        _register_library(container)

        _register_eval_recipe(container, recipe)

        _register_run(container, recipe, config, output_dir)

        _run_recipe(container)


@torch.inference_mode()
def generate(recipe: GenerationRecipe, config: object, output_dir: Path) -> None:
    from fairseq2.recipe.composition import _register_generation_recipe

    enable_deprecation_warnings()

    configure_rich_logging()

    container = DependencyContainer()

    with _swap_default_resolver(container):
        _register_library(container)

        _register_generation_recipe(container, recipe)

        _register_run(container, recipe, config, output_dir)

        _run_recipe(container)


def _run_recipe(resolver: DependencyResolver) -> None:
    # Prepare cluster environment.
    cluster_preparer = resolver.resolve(_ClusterPreparer)

    cluster_preparer.prepare()

    # Configure distributed logging.
    log_configurer = resolver.resolve(_DistributedLogConfigurer)

    log_configurer.configure()

    # Save config to file.
    config_dumper = resolver.resolve(_RecipeConfigDumper)

    config_dumper.dump()

    # Configure PyTorch.
    torch_configurer = resolver.resolve(_TorchConfigurer)

    torch_configurer.configure()

    # Log environment info.
    log_helper = resolver.resolve(_LogHelper)

    log_helper.log_system_info()

    log_helper.log_software_info()

    log_helper.log_environment_variables()

    # Run recipe task.
    task = resolver.resolve(Task)

    task_runner = resolver.resolve(_TaskRunner)

    task_runner.run(task)


@contextmanager
def _swap_default_resolver(resolver: DependencyResolver) -> Iterator[None]:
    original_resolver = fairseq2.runtime.dependency._resolver

    fairseq2.runtime.dependency._resolver = resolver

    try:
        yield
    finally:
        fairseq2.runtime.dependency._resolver = original_resolver


def _register_run(
    container: DependencyContainer, recipe: Recipe, config: object, output_dir: Path
) -> None:
    config_kls = recipe.config_kls

    if not isinstance(config, config_kls):
        raise TypeError(
            f"`config` must be of type `{config_kls}`, but is of type `{type(config)}` instead."
        )

    # Recipe Configuration
    def get_config(resolver: DependencyResolver) -> object:
        config_preparer = resolver.resolve(_RecipeConfigPreparer)

        return config_preparer.prepare(config_kls, config)

    container.register(RecipeConfig, get_config)

    # Recipe Output Directory
    def create_output_dir(resolver: DependencyResolver) -> Path:
        dir_creator = resolver.resolve(_OutputDirectoryCreator)

        return dir_creator.create(output_dir)

    container.register(Path, create_output_dir)


@final
class _RecipeConfigDumper:
    def __init__(
        self,
        config: RecipeConfig,
        output_dir: Path,
        world_info: WorldInfo,
        log_helper: _LogHelper,
        value_converter: ValueConverter,
        yaml_dumper: YamlDumper,
    ) -> None:
        self._config = config
        self._output_dir = output_dir
        self._world_info = world_info
        self._log_helper = log_helper
        self._value_converter = value_converter
        self._yaml_dumper = yaml_dumper

    def dump(self) -> None:
        untyped_config = self._config.as_(object)

        self._log_helper.log_config("Config", untyped_config)

        if self._world_info.rank != 0:
            return

        unstructured_config = self._value_converter.unstructure(untyped_config)

        file = self._output_dir.joinpath("config.yaml")

        try:
            self._yaml_dumper.dump(unstructured_config, file)
        except OSError as ex:
            raise_operational_system_error(ex)
