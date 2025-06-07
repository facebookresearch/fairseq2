# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from signal import SIGUSR1, signal
from types import FrameType

import fairseq2.runtime.dependency
from fairseq2.error import InfraError
from fairseq2.file_system import FileSystem
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.recipe.base import EvalRecipe, GenerationRecipe, TrainRecipe
from fairseq2.recipe.cluster import WorldInfo
from fairseq2.recipe.composition import (
    _register_eval_recipe,
    _register_generation_recipe,
    _register_train_recipe,
)
from fairseq2.recipe.config import get_output_dir, get_recipe_config
from fairseq2.recipe.logging import _configure_distributed_logging
from fairseq2.recipe.rng import _manual_seed
from fairseq2.recipe.torch import _configure_torch
from fairseq2.recipe.utils.log import log_config, log_environment_info
from fairseq2.runtime.composition import _register_library
from fairseq2.runtime.dependency import DependencyResolver, StandardDependencyContainer
from fairseq2.task import Task, TaskStopException
from fairseq2.utils.merge import to_mergeable
from fairseq2.utils.rich import configure_rich_logging
from fairseq2.utils.stopwatch import Stopwatch
from fairseq2.utils.structured import ValueConverter
from fairseq2.utils.validation import ObjectValidator
from fairseq2.utils.yaml import YamlDumper


def train(recipe: TrainRecipe, config: object, output_dir: Path) -> None:
    if not isinstance(config, recipe.config_kls):
        raise TypeError(
            f"`config` is expected to be of type `{recipe.config_kls}`, but is of type `{type(config)}` instead."
        )

    configure_rich_logging()

    container = StandardDependencyContainer()

    # Library
    _register_library(container)

    # Recipe
    _register_train_recipe(container, recipe)

    # Recipe Configuration
    def get_config(resolver: DependencyResolver) -> object:
        _validate_config(resolver, config)

        return config

    container.register(object, get_config, key="config")

    # Recipe Output Directory
    def get_output_dir(resolver: DependencyResolver) -> Path:
        return _resolve_output_dir(resolver, output_dir)

    container.register(Path, get_output_dir, key="output_dir")

    _run_recipe(container)


def evaluate(recipe: EvalRecipe, config: object, output_dir: Path) -> None:
    if not isinstance(config, recipe.config_kls):
        raise TypeError(
            f"`config` is expected to be of type `{recipe.config_kls}`, but is of type `{type(config)}` instead."
        )

    configure_rich_logging()

    container = StandardDependencyContainer()

    # Library
    _register_library(container)

    # Recipe
    _register_eval_recipe(container, recipe)

    # Recipe Configuration
    def get_config(resolver: DependencyResolver) -> object:
        _validate_config(resolver, config)

        return config

    container.register(object, get_config, key="config")

    # Recipe Output Directory
    def get_output_dir(resolver: DependencyResolver) -> Path:
        return _resolve_output_dir(resolver, output_dir)

    container.register(Path, get_output_dir, key="output_dir")

    _run_recipe(container)


def generate(recipe: GenerationRecipe, config: object, output_dir: Path) -> None:
    if not isinstance(config, recipe.config_kls):
        raise TypeError(
            f"`config` is expected to be of type `{recipe.config_kls}`, but is of type `{type(config)}` instead."
        )

    configure_rich_logging()

    container = StandardDependencyContainer()

    # Library
    _register_library(container)

    # Recipe
    _register_generation_recipe(container, recipe)

    # Recipe Configuration
    def get_config(resolver: DependencyResolver) -> object:
        _validate_config(resolver, config)

        return config

    container.register(object, get_config, key="config")

    # Recipe Output Directory
    def get_output_dir(resolver: DependencyResolver) -> Path:
        return _resolve_output_dir(resolver, output_dir)

    container.register(Path, get_output_dir, key="output_dir")

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
    _configure_distributed_logging(resolver)

    _save_config(resolver)

    _configure_torch(resolver)

    _manual_seed(resolver)

    log_environment_info(resolver)

    gangs = resolver.resolve(Gangs)

    wall_watch = resolver.resolve(Stopwatch)

    task = resolver.resolve(Task)

    log.info("Running on {} process(es).", gangs.root.size)

    # Use SIGUSR1 as the stop signal.
    def request_stop(signum: int, frame: FrameType | None) -> None:
        log.info("SIGUSR1 received. Requesting recipe to stop.")

        task.request_stop()

    original_signal_handler = signal(SIGUSR1, request_stop)

    try:
        task.run()
    except TaskStopException:
        elapsed_time = int(wall_watch.get_elapsed_time())

        if task.step_nr == 0:
            log.info("Task stopped after {:,} second(s)!", elapsed_time)
        else:
            log.info("Task stopped after {:,} second(s) at step {}!", elapsed_time, task.step_nr)  # fmt: skip

        raise
    except KeyboardInterrupt:
        elapsed_time = int(wall_watch.get_elapsed_time())

        if task.step_nr == 0:
            log.info("Task canceled after {:,} second(s)!", elapsed_time)
        else:
            log.info("Task canceled after {:,} second(s) at step {}!", elapsed_time, task.step_nr)  # fmt: skip

        raise
    else:
        elapsed_time = int(wall_watch.get_elapsed_time())

        if task.step_nr == 0:
            log.info("Task finished in {:,} second(s)!", elapsed_time)
        else:
            log.info("Task finished in {:,} second(s) after {} step(s)!", elapsed_time, task.step_nr)  # fmt: skip
    finally:
        task.close()

        signal(SIGUSR1, original_signal_handler)


def _save_config(resolver: DependencyResolver) -> None:
    yaml_dumper = resolver.resolve(YamlDumper)

    value_converter = resolver.resolve(ValueConverter)

    world_info = resolver.resolve(WorldInfo)

    recipe_config = get_recipe_config(resolver)

    output_dir = get_output_dir(resolver)

    unstructured_config = value_converter.unstructure(recipe_config)

    log_config("Config", unstructured_config)

    if world_info.rank != 0:
        return

    file = output_dir.joinpath("config.yaml")

    if isinstance(unstructured_config, dict):
        unstructured_config = to_mergeable(unstructured_config)

    try:
        yaml_dumper.dump(unstructured_config, file)
    except OSError as ex:
        raise InfraError(
            f"A system error has occurred while saving the recipe configuration to the '{file}' file. See the nested exception for details."
        ) from ex


def _validate_config(resolver: DependencyResolver, config: object) -> None:
    validator = resolver.resolve(ObjectValidator)

    validator.validate(config)


def _resolve_output_dir(resolver: DependencyResolver, output_dir: Path) -> Path:
    file_system = resolver.resolve(FileSystem)

    try:
        return file_system.resolve(output_dir)
    except OSError as ex:
        raise InfraError(
            f"A system error has occurred while resolving the '{output_dir}' directory. See the nested exception for details."
        ) from ex
