# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from fairseq2.assets import AssetMetadataSource
from fairseq2.checkpoint import CheckpointManager, StandardCheckpointManager
from fairseq2.error import raise_operational_system_error
from fairseq2.gang import GangError, Gangs, raise_operational_gang_error
from fairseq2.recipe.base import (
    EvalRecipe,
    GenerationRecipe,
    Recipe,
    RecipeContext,
    TrainRecipe,
)
from fairseq2.recipe.component import ComponentManager, _StandardComponentManager
from fairseq2.recipe.composition.beam_search import _register_beam_search
from fairseq2.recipe.composition.config import _register_config_sections
from fairseq2.recipe.composition.data_parallel import _register_data_parallel_wrappers
from fairseq2.recipe.composition.dataset import _register_dataset
from fairseq2.recipe.composition.device_stat import _register_device_stat
from fairseq2.recipe.composition.eval_model import _register_eval_model_loader
from fairseq2.recipe.composition.evaluator import _register_evaluator_factory
from fairseq2.recipe.composition.generator import _register_generator_factory
from fairseq2.recipe.composition.lr_schedulers import _register_lr_schedulers
from fairseq2.recipe.composition.metric_recorders import _register_metric_recorders
from fairseq2.recipe.composition.model import _register_train_model
from fairseq2.recipe.composition.optim import _register_optim
from fairseq2.recipe.composition.profilers import _register_profilers
from fairseq2.recipe.composition.sampling import _register_sampling
from fairseq2.recipe.composition.seq_generator import _register_seq_generators
from fairseq2.recipe.composition.tokenizer import _register_tokenizers
from fairseq2.recipe.composition.trainer import _register_trainer_factory
from fairseq2.recipe.config import (
    ReferenceModelSection,
)
from fairseq2.recipe.evaluator import Evaluator
from fairseq2.recipe.generator import Generator
from fairseq2.recipe.internal.asset_config import (
    _AssetConfigOverrider,
    _StandardAssetConfigOverrider,
)
from fairseq2.recipe.internal.assets import (
    _ExtraAssetMetadataSource,
    _ExtraModelMetadataSource,
)
from fairseq2.recipe.internal.cluster import _ClusterPreparer
from fairseq2.recipe.internal.config_preparer import (
    _RecipeConfigPreparer,
    _RecipeConfigStructurer,
    _StandardRecipeConfigStructurer,
)
from fairseq2.recipe.internal.eval_model import _EvalModelLoader
from fairseq2.recipe.internal.gang import (
    _log_ranks,
    _RecipeFSDPGangsFactory,
    _RecipeGangsFactory,
    _warmup_gangs,
)
from fairseq2.recipe.internal.log import _LogHelper, _StandardLogHelper
from fairseq2.recipe.internal.logging import _DistributedLogConfigurer
from fairseq2.recipe.internal.output_dir import _OutputDirectoryCreator
from fairseq2.recipe.internal.sweep_tag import (
    _StandardSweepTagGenerator,
    _SweepTagGenerator,
)
from fairseq2.recipe.internal.task import _TaskRunner
from fairseq2.recipe.internal.torch import _TorchConfigurer
from fairseq2.recipe.model import RecipeModel
from fairseq2.recipe.run import _RecipeConfigDumper
from fairseq2.recipe.task import Task
from fairseq2.recipe.trainer import Trainer
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver
from fairseq2.utils.stopwatch import Stopwatch


def _register_train_recipe(container: DependencyContainer, recipe: TrainRecipe) -> None:
    _register_recipe_common(container)

    # Recipe
    container.register_instance(Recipe, recipe)

    container.register_instance(TrainRecipe, recipe)

    # Trainer
    def create_trainer(resolver: DependencyResolver) -> Trainer:
        context = RecipeContext(resolver)

        try:
            return recipe.create_trainer(context)
        except OSError as ex:
            raise_operational_system_error(ex)
        except GangError as ex:
            raise_operational_gang_error(ex)

    container.register(Task, create_trainer)

    _register_data_parallel_wrappers(container)

    _register_lr_schedulers(container)

    _register_optim(container)

    _register_trainer_factory(container)

    _register_train_model(container)

    container.register_type(
        CheckpointManager, StandardCheckpointManager, singleton=True
    )

    # Gangs
    def create_gangs(resolver: DependencyResolver) -> Gangs:
        gangs_factory = resolver.resolve(_RecipeGangsFactory)

        gangs = gangs_factory.create()

        fsdp_gangs_factory = resolver.resolve(_RecipeFSDPGangsFactory)

        gangs = fsdp_gangs_factory.create(gangs)

        _warmup_gangs(gangs)

        _log_ranks(gangs)

        return gangs

    container.register(Gangs, create_gangs, singleton=True)

    container.register_type(_RecipeFSDPGangsFactory)

    # Custom Objects
    recipe.register(container)


def _register_eval_recipe(container: DependencyContainer, recipe: EvalRecipe) -> None:
    _register_recipe_common(container)

    # Recipe
    container.register_instance(Recipe, recipe)

    # Model
    def load_model(resolver: DependencyResolver) -> RecipeModel:
        section = resolver.resolve(ReferenceModelSection)

        model_loader = resolver.resolve(_EvalModelLoader)

        return model_loader.load("model", section)

    container.register(RecipeModel, load_model, singleton=True)

    # Evaluator
    @torch.inference_mode()
    def create_evaluator(resolver: DependencyResolver) -> Evaluator:
        context = RecipeContext(resolver)

        try:
            return recipe.create_evaluator(context)
        except OSError as ex:
            raise_operational_system_error(ex)
        except GangError as ex:
            raise_operational_gang_error(ex)

    container.register(Task, create_evaluator)

    # Gangs
    def create_gangs(resolver: DependencyResolver) -> Gangs:
        gangs_factory = resolver.resolve(_RecipeGangsFactory)

        gangs = gangs_factory.create()

        _warmup_gangs(gangs)

        _log_ranks(gangs)

        return gangs

    container.register(Gangs, create_gangs, singleton=True)

    # Custom Objects
    recipe.register(container)


def _register_generation_recipe(
    container: DependencyContainer, recipe: GenerationRecipe
) -> None:
    _register_recipe_common(container)

    # Recipe
    container.register_instance(Recipe, recipe)

    # Model
    def load_model(resolver: DependencyResolver) -> RecipeModel:
        section = resolver.resolve(ReferenceModelSection)

        model_loader = resolver.resolve(_EvalModelLoader)

        return model_loader.load("model", section)

    container.register(RecipeModel, load_model, singleton=True)

    # Generator
    @torch.inference_mode()
    def create_generator(resolver: DependencyResolver) -> Generator:
        context = RecipeContext(resolver)

        try:
            return recipe.create_generator(context)
        except OSError as ex:
            raise_operational_system_error(ex)
        except GangError as ex:
            raise_operational_gang_error(ex)

    container.register(Task, create_generator)

    # Gangs
    def create_gangs(resolver: DependencyResolver) -> Gangs:
        gangs_factory = resolver.resolve(_RecipeGangsFactory)

        gangs = gangs_factory.create()

        _warmup_gangs(gangs)

        _log_ranks(gangs)

        return gangs

    container.register(Gangs, create_gangs, singleton=True)

    # Custom Objects
    recipe.register(container)


def _register_recipe_common(container: DependencyContainer) -> None:
    # Wall Watch
    wall_watch = Stopwatch()

    wall_watch.start()

    container.register_instance(Stopwatch, wall_watch)

    # wire_object
    _register_beam_search(container)
    _register_config_sections(container)
    _register_dataset(container)
    _register_device_stat(container)
    _register_eval_model_loader(container)
    _register_evaluator_factory(container)
    _register_generator_factory(container)
    _register_metric_recorders(container)
    _register_profilers(container)
    _register_sampling(container)
    _register_seq_generators(container)
    _register_tokenizers(container)

    container.register_type(_AssetConfigOverrider, _StandardAssetConfigOverrider)
    container.register_type(_ClusterPreparer)
    container.register_type(ComponentManager, _StandardComponentManager, singleton=True)
    container.register_type(_DistributedLogConfigurer)
    container.register_type(_LogHelper, _StandardLogHelper)
    container.register_type(_OutputDirectoryCreator)
    container.register_type(_RecipeConfigDumper)
    container.register_type(_RecipeConfigPreparer)
    container.register_type(_RecipeConfigStructurer, _StandardRecipeConfigStructurer)
    container.register_type(_RecipeGangsFactory)
    container.register_type(_SweepTagGenerator, _StandardSweepTagGenerator)
    container.register_type(_TaskRunner)
    container.register_type(_TorchConfigurer)

    container.collection.register_type(AssetMetadataSource, _ExtraAssetMetadataSource)
    container.collection.register_type(AssetMetadataSource, _ExtraModelMetadataSource)
