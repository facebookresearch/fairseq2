# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from signal import SIGUSR1, signal
from types import FrameType

from fairseq2.composition import register_library
from fairseq2.dependency import DependencyResolver, StandardDependencyContainer
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.recipe.base import EvalRecipe, GenerationRecipe, TrainRecipe
from fairseq2.recipe.composition import (
    register_eval_recipe,
    register_generation_recipe,
    register_train_recipe,
)
from fairseq2.recipe.config import save_config
from fairseq2.recipe.logging import setup_distributed_logging, setup_logging
from fairseq2.recipe.rng import set_manual_rng_seed
from fairseq2.recipe.task import Task, TaskStopException
from fairseq2.recipe.torch import set_torch_distributed_variables, setup_torch
from fairseq2.utils.stopwatch import Stopwatch


def train(recipe: TrainRecipe, config: object, output_dir: Path) -> None:
    if not isinstance(config, recipe.config_kls):
        raise TypeError(
            f"`config` is expected to be of type `{recipe.config_kls}`, but is of type `{type(config)}` instead."
        )

    container = StandardDependencyContainer()

    # Library
    register_library(container)

    # Recipe
    register_train_recipe(container, recipe)

    # Recipe Configuration
    container.register_instance(object, config, key="config")

    # Recipe Output Directory
    container.register_instance(Path, output_dir, key="output_dir")

    run_recipe(container)


def evaluate(recipe: EvalRecipe, config: object, output_dir: Path) -> None:
    if not isinstance(config, recipe.config_kls):
        raise TypeError(
            f"`config` is expected to be of type `{recipe.config_kls}`, but is of type `{type(config)}` instead."
        )

    container = StandardDependencyContainer()

    # Library
    register_library(container)

    # Recipe
    register_eval_recipe(container, recipe)

    # Recipe Configuration
    container.register_instance(object, config, key="config")

    # Recipe Output Directory
    container.register_instance(Path, output_dir, key="output_dir")

    run_recipe(container)


def generate(recipe: GenerationRecipe, config: object, output_dir: Path) -> None:
    if not isinstance(config, recipe.config_kls):
        raise TypeError(
            f"`config` is expected to be of type `{recipe.config_kls}`, but is of type `{type(config)}` instead."
        )

    container = StandardDependencyContainer()

    # Library
    register_library(container)

    # Recipe
    register_generation_recipe(container, recipe)

    # Recipe Configuration
    container.register_instance(object, config, key="config")

    # Recipe Output Directory
    container.register_instance(Path, output_dir, key="output_dir")

    run_recipe(container)


def run_recipe(resolver: DependencyResolver) -> None:
    setup_logging(resolver)

    set_torch_distributed_variables(resolver)

    setup_distributed_logging(resolver)

    save_config(resolver)

    setup_torch(resolver)

    set_manual_rng_seed(resolver)

    gangs = resolver.resolve(Gangs)

    log.info("Running on {} process(es).", gangs.root.size)

    wall_watch = resolver.resolve(Stopwatch)

    task = resolver.resolve(Task)

    # Use SIGUSR1 as the stop signal.
    def request_stop(signum: int, frame: FrameType | None) -> None:
        log.info("SIGUSR1 received. Requesting recipe to stop.")

        task.request_stop()

    signal(SIGUSR1, request_stop)

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
