# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from fairseq2.assets import AssetMetadataSource
from fairseq2.checkpoint import (
    CheckpointManager,
    StandardCheckpointManager,
    _ModelMetadataDumper,
    _ModelMetadataLoader,
    _StandardModelMetadataDumper,
    _StandardModelMetadataLoader,
)
from fairseq2.device import (
    CPU,
    CudaContext,
    Device,
    _DefaultDeviceDetector,
    _StandardCudaContext,
)
from fairseq2.error import OperationalError
from fairseq2.gang import GangError, Gangs
from fairseq2.recipe.base import Recipe, RecipeContext
from fairseq2.recipe.component import ComponentManager, _StandardComponentManager
from fairseq2.recipe.composition.beam_search import _register_beam_search
from fairseq2.recipe.composition.config import _register_config_sections
from fairseq2.recipe.composition.data_parallel import _register_data_parallel_wrappers
from fairseq2.recipe.composition.dataset import _register_default_dataset
from fairseq2.recipe.composition.device_stat import _register_device_stat
from fairseq2.recipe.composition.evaluator import _register_evaluator_factory
from fairseq2.recipe.composition.generator import _register_generator_factory
from fairseq2.recipe.composition.lr_schedulers import _register_lr_schedulers
from fairseq2.recipe.composition.metric_recorders import _register_metric_recorders
from fairseq2.recipe.composition.optim import _register_optim
from fairseq2.recipe.composition.profilers import _register_profilers
from fairseq2.recipe.composition.reference_model import (
    _register_inference_model,
    _register_reference_model_loader,
)
from fairseq2.recipe.composition.sampling import _register_sampling
from fairseq2.recipe.composition.seq_generator import _register_seq_generators
from fairseq2.recipe.composition.tokenizer import _register_default_tokenizer
from fairseq2.recipe.composition.train_model import _register_train_model
from fairseq2.recipe.composition.trainer import _register_trainer_factory
from fairseq2.recipe.error import ConfigError
from fairseq2.recipe.internal.asset_config import (
    _AssetConfigOverrider,
    _StandardAssetConfigOverrider,
)
from fairseq2.recipe.internal.assets import (
    _ExtraAssetMetadataSource,
    _ExtraModelMetadataSource,
)
from fairseq2.recipe.internal.cluster import _ClusterPreparer
from fairseq2.recipe.internal.gang import (
    _FSDPGangsFactory,
    _GangsFactory,
    _warmup_gangs,
)
from fairseq2.recipe.internal.hook import _HookManager, _TrainHookManager
from fairseq2.recipe.internal.log import _log_ranks, _LogHelper, _StandardLogHelper
from fairseq2.recipe.internal.logging import _DistributedLogConfigurer
from fairseq2.recipe.internal.output_dir import _OutputDirectoryCreator
from fairseq2.recipe.internal.sweep_tag import (
    _StandardSweepTagGenerator,
    _SweepTagGenerator,
)
from fairseq2.recipe.internal.task import _TaskRunner
from fairseq2.recipe.internal.torch import _TorchConfigurer
from fairseq2.recipe.run import _RecipeConfigDumper
from fairseq2.recipe.task import Task
from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyResolver,
    wire_object,
)
from fairseq2.utils.env import Environment, EnvironmentVariableError
from fairseq2.utils.rng import RngBag
from fairseq2.utils.stopwatch import Stopwatch
from fairseq2.utils.threading import ThreadPool, _StandardThreadPool
from fairseq2.world_info import WorldInfo


def _register_train_recipe(container: DependencyContainer, recipe: Recipe) -> None:
    _register_recipe_common(container)

    container.register_instance(Recipe, recipe)

    _register_train_model(container)

    # Task
    def create_task(resolver: DependencyResolver) -> Task:
        context = RecipeContext(resolver)

        try:
            return recipe.create_task(context)
        except (RuntimeError, OSError, GangError) as ex:
            raise OperationalError("Failed to create task.") from ex

    container.register(Task, create_task)

    # Gangs
    def create_gangs(resolver: DependencyResolver) -> Gangs:
        gangs_factory = resolver.resolve(_GangsFactory)

        gangs = gangs_factory.create()

        fsdp_gangs_factory = resolver.resolve(_FSDPGangsFactory)

        gangs = fsdp_gangs_factory.create(gangs)

        _warmup_gangs(gangs)

        _log_ranks(gangs)

        return gangs

    container.register(Gangs, create_gangs, singleton=True)

    container.register_type(_FSDPGangsFactory)

    container.register_type(
        CheckpointManager, StandardCheckpointManager, singleton=True
    )

    container.register_type(_TrainHookManager, singleton=True)

    _register_data_parallel_wrappers(container)
    _register_lr_schedulers(container)
    _register_optim(container)
    _register_trainer_factory(container)

    # Custom Objects
    recipe.register(container)


def _register_inference_recipe(container: DependencyContainer, recipe: Recipe) -> None:
    _register_recipe_common(container)

    container.register_instance(Recipe, recipe)

    _register_inference_model(container)

    # Task
    @torch.inference_mode()
    def create_task(resolver: DependencyResolver) -> Task:
        context = RecipeContext(resolver)

        try:
            return recipe.create_task(context)
        except (RuntimeError, OSError, GangError) as ex:
            raise OperationalError("Failed to create task.") from ex

    container.register(Task, create_task)

    # Gangs
    def create_gangs(resolver: DependencyResolver) -> Gangs:
        gangs_factory = resolver.resolve(_GangsFactory)

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

    _register_beam_search(container)
    _register_config_sections(container)
    _register_default_dataset(container)
    _register_default_tokenizer(container)
    _register_device_stat(container)
    _register_evaluator_factory(container)
    _register_generator_factory(container)
    _register_metric_recorders(container)
    _register_profilers(container)
    _register_reference_model_loader(container)
    _register_sampling(container)
    _register_seq_generators(container)

    # Device
    def detect_default_device(resolver: DependencyResolver) -> Device:
        device_detector = resolver.resolve(_DefaultDeviceDetector)

        try:
            return device_detector.detect()
        except EnvironmentVariableError as ex:
            raise ConfigError(f"Default device cannot be detected. {ex}") from None

    container.register(Device, detect_default_device, singleton=True)

    # WorldInfo
    def get_world_info(resolver: DependencyResolver) -> WorldInfo:
        env = resolver.resolve(Environment)

        try:
            return WorldInfo.from_env(env)
        except EnvironmentVariableError as ex:
            raise ConfigError(f"{ex}") from None

    container.register(WorldInfo, get_world_info, singleton=True)

    # ThreadPool
    def create_thread_pool(resolver: DependencyResolver) -> ThreadPool:
        world_info = resolver.resolve(WorldInfo)

        try:
            return _StandardThreadPool.create_default(world_info.local_size)
        except RuntimeError as ex:
            raise OperationalError(
                "Failed to initialize the thread pool of the process."
            ) from ex

    container.register(ThreadPool, create_thread_pool, singleton=True)

    # RngBag
    def create_rng_bag(resolver: DependencyResolver) -> RngBag:
        device = resolver.resolve(Device)

        return RngBag.from_device_defaults(CPU, device)

    container.register(RngBag, create_rng_bag, singleton=True)

    # TaskRunner
    def create_task_runner(resolver: DependencyResolver) -> _TaskRunner:
        task_runner = wire_object(resolver, _TaskRunner)

        hook_manager = resolver.resolve(_HookManager)

        hook_manager.maybe_register_task_hooks(task_runner)

        return task_runner

    container.register(_TaskRunner, create_task_runner)

    container.register_type(_AssetConfigOverrider, _StandardAssetConfigOverrider)
    container.register_type(_ClusterPreparer)
    container.register_type(ComponentManager, _StandardComponentManager, singleton=True)
    container.register_type(CudaContext, _StandardCudaContext)
    container.register_type(_DefaultDeviceDetector)
    container.register_type(_DistributedLogConfigurer)
    container.register_type(_GangsFactory)
    container.register_type(_HookManager, singleton=True)
    container.register_type(_LogHelper, _StandardLogHelper)
    container.register_type(_ModelMetadataDumper, _StandardModelMetadataDumper)
    container.register_type(_ModelMetadataLoader, _StandardModelMetadataLoader)
    container.register_type(_OutputDirectoryCreator)
    container.register_type(_RecipeConfigDumper)
    container.register_type(_SweepTagGenerator, _StandardSweepTagGenerator)
    container.register_type(_TorchConfigurer)

    container.collection.register_type(AssetMetadataSource, _ExtraAssetMetadataSource)
    container.collection.register_type(AssetMetadataSource, _ExtraModelMetadataSource)
