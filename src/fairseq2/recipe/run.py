# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import fairseq2.runtime.dependency
from fairseq2.composition import _register_library
from fairseq2.device import Device
from fairseq2.recipe.base import EvalRecipe, GenerationRecipe, TrainRecipe
from fairseq2.recipe.composition import (
    _register_eval_recipe,
    _register_generation_recipe,
    _register_train_recipe,
)
from fairseq2.recipe.recipe_preparer import RecipePreparer
from fairseq2.recipe.task import TaskRunner
from fairseq2.recipe.torch import TorchConfigurer
from fairseq2.recipe.wire import _RecipeConfigPreparer
from fairseq2.runtime.dependency import DependencyResolver, StandardDependencyContainer
from fairseq2.utils.log import (
    log_environment_variables,
    log_software_info,
    log_system_info,
)
from fairseq2.utils.rich import configure_rich_logging


def train(recipe: TrainRecipe, config: object, output_dir: Path) -> None:
    if not isinstance(config, recipe.config_kls):
        raise TypeError(
            f"`config` must be of type `{recipe.config_kls}`, but is of type `{type(config)}` instead."
        )

    configure_rich_logging()

    container = StandardDependencyContainer()

    # Library
    _register_library(container)

    # Recipe
    _register_train_recipe(container, recipe)

    # Recipe Configuration
    def get_config(resolver: DependencyResolver) -> object:
        return _RecipeConfigPreparer(resolver).prepare(config)

    container.register(object, get_config, key="config")

    # Recipe Output Directory
    def create_output_dir(resolver: DependencyResolver) -> Path:
        return resolver.resolve(RecipePreparer).prepare(output_dir)

    container.register(Path, create_output_dir)

    _run_recipe(container)


def evaluate(recipe: EvalRecipe, config: object, output_dir: Path) -> None:
    if not isinstance(config, recipe.config_kls):
        raise TypeError(
            f"`config` must be of type `{recipe.config_kls}`, but is of type `{type(config)}` instead."
        )

    configure_rich_logging()

    container = StandardDependencyContainer()

    # Library
    _register_library(container)

    # Recipe
    _register_eval_recipe(container, recipe)

    # Recipe Configuration
    def get_config(resolver: DependencyResolver) -> object:
        return _RecipeConfigPreparer(resolver).prepare(config)

    container.register(object, get_config, key="config")

    # Recipe Output Directory
    def create_output_dir(resolver: DependencyResolver) -> Path:
        return resolver.resolve(RecipePreparer).prepare(output_dir)

    container.register(Path, create_output_dir)

    _run_recipe(container)


def generate(recipe: GenerationRecipe, config: object, output_dir: Path) -> None:
    if not isinstance(config, recipe.config_kls):
        raise TypeError(
            f"`config` must be of type `{recipe.config_kls}`, but is of type `{type(config)}` instead."
        )

    configure_rich_logging()

    container = StandardDependencyContainer()

    # Library
    _register_library(container)

    # Recipe
    _register_generation_recipe(container, recipe)

    # Recipe Configuration
    def get_config(resolver: DependencyResolver) -> object:
        return _RecipeConfigPreparer(resolver).prepare(config)

    container.register(object, get_config, key="config")

    # Recipe Output Directory
    def create_output_dir(resolver: DependencyResolver) -> Path:
        return resolver.resolve(RecipePreparer).prepare(output_dir)

    container.register(Path, create_output_dir)

    _run_recipe(container)


def _run_recipe(resolver: DependencyResolver) -> None:
    with _swap_default_resolver(resolver):
        _do_run_recipe(resolver)


@contextmanager
def _swap_default_resolver(resolver: DependencyResolver) -> Iterator[None]:
    original_resolver = fairseq2.runtime.dependency._resolver

    fairseq2.runtime.dependency._resolver = resolver

    try:
        yield
    finally:
        fairseq2.runtime.dependency._resolver = original_resolver


def _do_run_recipe(resolver: DependencyResolver) -> None:
    torch_configurer = resolver.resolve(TorchConfigurer)
    torch_configurer.configure()

    device = resolver.resolve(Device)

    log_system_info(device)

    log_software_info()

    log_environment_variables()

    task_runner = resolver.resolve(TaskRunner)

    task_runner.run()
