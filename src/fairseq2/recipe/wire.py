# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

from torch.optim import Optimizer

from fairseq2.assets import AssetStore, FileAssetMetadataLoader
from fairseq2.checkpoint import (
    CheckpointManager,
    ModelMetadataDumper,
    ModelMetadataLoader,
    StandardCheckpointManager,
)
from fairseq2.cluster import ClusterResolver
from fairseq2.data.tokenizers import Tokenizer, TokenizerFamilyHandler
from fairseq2.datasets import DatasetFamilyHandler
from fairseq2.device import Device
from fairseq2.file_system import FileSystem
from fairseq2.gang import Gangs
from fairseq2.io import TensorDumper, TensorLoader
from fairseq2.metrics.recorders import (
    CompositeMetricRecorder,
    JsonlMetricRecorder,
    LogMetricRecorder,
    MetricDescriptor,
    MetricRecorder,
    TensorBoardRecorder,
    WandbClient,
    WandbRecorder,
)
from fairseq2.model import Model
from fairseq2.models import ModelFamilyHandler
from fairseq2.profilers import CompositeProfiler, Profiler
from fairseq2.recipe.asset_config import (
    AssetConfigOverrider,
    StandardAssetConfigOverrider,
)
from fairseq2.recipe.assets import (
    MaybeExtraAssetMetadataSource,
    MaybeExtraModelMetadataSource,
)
from fairseq2.recipe.base import Recipe, TrainRecipe
from fairseq2.recipe.beam_search import BeamSearchFactory
from fairseq2.recipe.component import ComponentManager, StandardComponentManager
from fairseq2.recipe.config import (
    CommonSection,
    DatasetSection,
    LRSchedulerSection,
    ModelSection,
    OptimizerSection,
    RegimeSection,
    Seq2SeqGeneratorSection,
    SequenceGeneratorSection,
    TrainerSection,
    get_config_section,
)
from fairseq2.recipe.config_preparer import (
    RecipeConfigPreparer,
    RecipeConfigStructurer,
    StandardRecipeConfigStructurer,
)
from fairseq2.recipe.data_parallel import (
    DDPFactory,
    DDPModelWrapper,
    DelegatingDPModelWrapper,
    DPModelWrapper,
    FSDPFactory,
    FSDPModelWrapper,
)
from fairseq2.recipe.dataset import DatasetFactory
from fairseq2.recipe.device_stat import DeviceStatTrackerFactory
from fairseq2.recipe.eval_model import (
    DelegatingEvalModelPreparer,
    EvalModelBootstrapper,
    EvalModelFactory,
    EvalModelPreparer,
    RecipeEvalModelPreparer,
    StandardEvalModelBootstrapper,
    StandardEvalModelPreparer,
)
from fairseq2.recipe.logging import (
    DistributedLogConfigurer,
    StandardDistributedLogConfigurer,
)
from fairseq2.recipe.lr_schedulers import (
    CosineAnnealingLRFactory,
    LRSchedulerFactory,
    MyleLRFactory,
    NoamLRFactory,
    PolynomialDecayLRFactory,
    TriStageLRFactory,
)
from fairseq2.recipe.metric_recorders import (
    MaybeTensorBoardRecorderFactory,
    MaybeWandbRecorderFactory,
    MetricRecorderFactory,
    StandardWandbRunIdManager,
    WandbClientFactory,
    WandbIdGenerator,
    WandbRunFactory,
    WandbRunIdManager,
)
from fairseq2.recipe.model import (
    DelegatingModelPreparer,
    ModelBootstrapper,
    ModelFactory,
    ModelMetadataSaver,
    ModelPreparer,
    RecipeModelPreparer,
    StandardModelBootstrapper,
    StandardModelMetadataSaver,
    StandardModelPreparer,
)
from fairseq2.recipe.optim import AdafactorFactory, AdamWFactory, OptimizerFactory
from fairseq2.recipe.profilers import MaybeTorchProfilerFactory
from fairseq2.recipe.recipe_preparer import (
    OutputDirectoryCreator,
    RecipeConfigDumper,
    RecipePreparer,
    StandardOutputDirectoryCreator,
    StandardRecipeConfigDumper,
)
from fairseq2.recipe.sampling import SamplingFactory
from fairseq2.recipe.seed import SeedHolder
from fairseq2.recipe.seq_generator import (
    Seq2SeqGeneratorFactory,
    SequenceGeneratorFactory,
)
from fairseq2.recipe.sweep_tag import StandardSweepTagGenerator, SweepTagGenerator
from fairseq2.recipe.task import TaskRunner
from fairseq2.recipe.tokenizer import TokenizerFactory
from fairseq2.recipe.torch import StandardTorchConfigurer, TorchConfigurer
from fairseq2.recipe.world import WorldPreparer
from fairseq2.runtime.dependency import DependencyProvider, DependencyResolver
from fairseq2.task import Task
from fairseq2.utils.config import ConfigMerger
from fairseq2.utils.device_stat import DeviceStatTracker
from fairseq2.utils.env import Environment
from fairseq2.utils.rng import RngBag
from fairseq2.utils.stopwatch import Stopwatch
from fairseq2.utils.structured import ValueConverter
from fairseq2.utils.threading import ThreadPool
from fairseq2.utils.validation import ObjectValidator
from fairseq2.utils.yaml import YamlDumper
from fairseq2.world_info import WorldInfo

# fmt: off


def _AdafactorFactory(resolver: DependencyResolver) -> AdafactorFactory:
    model = resolver.resolve(Model)

    return AdafactorFactory(model)


def _AdamWFactory(resolver: DependencyResolver) -> AdamWFactory:
    model = resolver.resolve(Model)

    return AdamWFactory(model)


def _BeamSearchFactory(resolver: DependencyResolver) -> BeamSearchFactory:
    model = resolver.resolve(Model)

    tokenizer = resolver.resolve(Tokenizer)

    component_manager = resolver.resolve(ComponentManager)

    return BeamSearchFactory(model, tokenizer, component_manager)


def _CompositeMetricRecorder(resolver: DependencyResolver) -> MetricRecorder:
    recorders = resolver.resolve_all(MetricRecorder, key="alt")

    return CompositeMetricRecorder(recorders)


def _CompositeProfiler(resolver: DependencyResolver) -> Profiler:
    profilers = resolver.resolve_all(Profiler, key="alt")

    return CompositeProfiler(profilers)


def _RecipeConfigPreparer(resolver: DependencyResolver) -> RecipeConfigPreparer:
    structurer = resolver.resolve(RecipeConfigStructurer)

    validator = resolver.resolve(ObjectValidator)

    return RecipeConfigPreparer(structurer, validator)


def _CosineAnnealingLRFactory(resolver: DependencyResolver) -> CosineAnnealingLRFactory:
    optimizer = resolver.resolve(Optimizer)

    regime_section = get_config_section(resolver, "regime", RegimeSection)

    return CosineAnnealingLRFactory(optimizer, regime_section)


def _DatasetFactory(resolver: DependencyResolver) -> DatasetFactory:
    section = get_config_section(resolver, "dataset", DatasetSection)

    handlers = DependencyProvider(resolver, DatasetFamilyHandler)

    asset_store = resolver.resolve(AssetStore)

    asset_config_overrider = resolver.resolve(AssetConfigOverrider)

    gangs = resolver.resolve(Gangs)

    return DatasetFactory(
        section, handlers, asset_store, asset_config_overrider, gangs
    )


def _DDPModelWrapper(resolver: DependencyResolver, factory: DDPFactory, static_graph: bool) -> DPModelWrapper:
    gangs = resolver.resolve(Gangs)

    return DDPModelWrapper(factory, gangs, static_graph)


def _DelegatingDPModelWrapper(resolver: DependencyResolver) -> DPModelWrapper:
    wrappers = DependencyProvider(resolver, DPModelWrapper)

    section = get_config_section(resolver, "trainer", TrainerSection)

    gangs = resolver.resolve(Gangs)

    return DelegatingDPModelWrapper(wrappers, section, gangs)


def _DelegatingEvalModelPreparer(resolver: DependencyResolver) -> EvalModelPreparer:
    preparers = resolver.resolve_all(EvalModelPreparer, key="alt")

    return DelegatingEvalModelPreparer(preparers)


def _DelegatingModelPreparer(resolver: DependencyResolver) -> ModelPreparer:
    preparers = resolver.resolve_all(ModelPreparer, key="alt")

    return DelegatingModelPreparer(preparers)


def _DeviceStatTrackerFactory(resolver: DependencyResolver) -> DeviceStatTrackerFactory:
    trackers = DependencyProvider(resolver, DeviceStatTracker)

    gangs = resolver.resolve(Gangs)

    return DeviceStatTrackerFactory(trackers, gangs)


def _EvalModelFactory(resolver: DependencyResolver) -> EvalModelFactory:
    bootstrapper = resolver.resolve(EvalModelBootstrapper)

    preparer = resolver.resolve(EvalModelPreparer)

    gangs = resolver.resolve(Gangs)

    return EvalModelFactory(bootstrapper, preparer, gangs)


def _MaybeExtraAssetMetadataSource(resolver: DependencyResolver) -> MaybeExtraAssetMetadataSource:
    section = get_config_section(resolver, "common", CommonSection)

    metadata_loader = resolver.resolve(FileAssetMetadataLoader)

    return MaybeExtraAssetMetadataSource(section, metadata_loader)


def _MaybeExtraModelMetadataSource(resolver: DependencyResolver) -> MaybeExtraModelMetadataSource:
    section = get_config_section(resolver, "common", CommonSection)

    metadata_loader = resolver.resolve(ModelMetadataLoader)

    return MaybeExtraModelMetadataSource(section, metadata_loader)


def _FSDPModelWrapper(resolver: DependencyResolver, factory: FSDPFactory) -> DPModelWrapper:
    section = get_config_section(resolver, "trainer", TrainerSection)

    checkpoint_manager = resolver.resolve(CheckpointManager)

    gangs = resolver.resolve(Gangs)

    return FSDPModelWrapper(factory, section, checkpoint_manager, gangs)


def _JsonlMetricRecorder(resolver: DependencyResolver) -> MetricRecorder:
    output_dir = resolver.resolve(Path)

    file_system = resolver.resolve(FileSystem)

    metric_descriptors = DependencyProvider(resolver, MetricDescriptor)

    return JsonlMetricRecorder(output_dir, file_system, metric_descriptors)


def _LogMetricRecorder(resolver: DependencyResolver) -> MetricRecorder:
    metric_descriptors = DependencyProvider(resolver, MetricDescriptor)

    return LogMetricRecorder(metric_descriptors)


def _LRSchedulerFactory(resolver: DependencyResolver) -> LRSchedulerFactory:
    section = get_config_section(
        resolver, "lr_scheduler", LRSchedulerSection, allow_none=True
    )

    optimizer = resolver.resolve(Optimizer)

    component_manager = resolver.resolve(ComponentManager)

    return LRSchedulerFactory(section, optimizer, component_manager)


def _MetricRecorderFactory(resolver: DependencyResolver) -> MetricRecorderFactory:
    gangs = resolver.resolve(Gangs)

    default_factory = lambda: resolver.resolve(CompositeMetricRecorder)

    return MetricRecorderFactory(gangs, default_factory)


def _ModelFactory(resolver: DependencyResolver) -> ModelFactory:
    bootstrapper = resolver.resolve(ModelBootstrapper)

    metadata_saver = resolver.resolve(ModelMetadataSaver)

    preparer = resolver.resolve(ModelPreparer)

    gangs = resolver.resolve(Gangs)

    return ModelFactory(bootstrapper, metadata_saver, preparer, gangs)


def _MyleLRFactory(resolver: DependencyResolver) -> MyleLRFactory:
    optimizer = resolver.resolve(Optimizer)

    return MyleLRFactory(optimizer)


def _NoamLRFactory(resolver: DependencyResolver) -> NoamLRFactory:
    optimizer = resolver.resolve(Optimizer)

    return NoamLRFactory(optimizer)


def _OptimizerFactory(resolver: DependencyResolver) -> OptimizerFactory:
    section = get_config_section(resolver, "optimizer", OptimizerSection)

    trainer_section = get_config_section(resolver, "trainer", TrainerSection)

    component_manager = resolver.resolve(ComponentManager)

    gangs = resolver.resolve(Gangs)

    return OptimizerFactory(section, trainer_section, component_manager, gangs)


def _RecipePreparer(resolver: DependencyResolver) -> RecipePreparer:
    dir_creator = resolver.resolve(OutputDirectoryCreator)

    dist_log_configurer = resolver.resolve(DistributedLogConfigurer)

    config_dumper = resolver.resolve(RecipeConfigDumper)

    return RecipePreparer(dir_creator, dist_log_configurer, config_dumper)


def _PolynomialDecayLRFactory(resolver: DependencyResolver) -> PolynomialDecayLRFactory:
    optimizer = resolver.resolve(Optimizer)

    regime_section = get_config_section(resolver, "regime", RegimeSection)

    return PolynomialDecayLRFactory(optimizer, regime_section)


def _RecipeEvalModelPreparer(resolver: DependencyResolver) -> EvalModelPreparer:
    recipe = resolver.resolve(Recipe)

    return RecipeEvalModelPreparer(recipe, resolver)


def _RecipeModelPreparer(resolver: DependencyResolver) -> ModelPreparer:
    recipe = resolver.resolve(TrainRecipe)

    return RecipeModelPreparer(recipe, resolver)


def _SamplingFactory(resolver: DependencyResolver) -> SamplingFactory:
    model = resolver.resolve(Model)

    tokenizer = resolver.resolve(Tokenizer)

    component_manager = resolver.resolve(ComponentManager)

    return SamplingFactory(model, tokenizer, component_manager)


def _SeedHolder(resolver: DependencyResolver) -> SeedHolder:
    section = get_config_section(resolver, "common", CommonSection)

    return SeedHolder(section)


def _Seq2SeqGeneratorFactory(resolver: DependencyResolver) -> Seq2SeqGeneratorFactory:
    section = get_config_section(resolver, "seq2seq_generator", Seq2SeqGeneratorSection)

    component_manager = resolver.resolve(ComponentManager)

    return Seq2SeqGeneratorFactory(section, component_manager)


def _SequenceGeneratorFactory(resolver: DependencyResolver) -> SequenceGeneratorFactory:
    section = get_config_section(resolver, "seq_generator", SequenceGeneratorSection)

    component_manager = resolver.resolve(ComponentManager)

    return SequenceGeneratorFactory(section, component_manager)


def _StandardAssetConfigOverrider(resolver: DependencyResolver) -> AssetConfigOverrider:
    value_converter = resolver.resolve(ValueConverter)

    config_merger = resolver.resolve(ConfigMerger)

    validator = resolver.resolve(ObjectValidator)

    return StandardAssetConfigOverrider(value_converter, config_merger, validator)


def _StandardCheckpointManager(resolver: DependencyResolver) -> CheckpointManager:
    output_dir = resolver.resolve(Path)

    gangs = resolver.resolve(Gangs)

    file_system = resolver.resolve(FileSystem)

    tensor_loader = resolver.resolve(TensorLoader)

    tensor_dumper = resolver.resolve(TensorDumper)

    thread_pool = resolver.resolve(ThreadPool)

    return StandardCheckpointManager(
        output_dir, gangs, file_system, tensor_loader, tensor_dumper, thread_pool
    )


def _StandardComponentManager(resolver: DependencyResolver) -> ComponentManager:
    value_converter = resolver.resolve(ValueConverter)

    return StandardComponentManager(resolver, value_converter)


def _StandardRecipeConfigDumper(resolver: DependencyResolver) -> RecipeConfigDumper:
    config = resolver.resolve(object, key="config")

    world_info = resolver.resolve(WorldInfo)

    value_converter = resolver.resolve(ValueConverter)

    yaml_dumper = resolver.resolve(YamlDumper)

    return StandardRecipeConfigDumper(config, world_info, value_converter, yaml_dumper)


def _StandardRecipeConfigStructurer(resolver: DependencyResolver) -> RecipeConfigStructurer:
    config_kls = resolver.resolve(type, key="config_kls")

    component_manager = resolver.resolve(ComponentManager)

    value_converter = resolver.resolve(ValueConverter)

    return StandardRecipeConfigStructurer(
        config_kls, component_manager, value_converter, resolver
    )


def _StandardEvalModelBootstrapper(resolver: DependencyResolver) -> StandardEvalModelBootstrapper:
    handlers = DependencyProvider(resolver, ModelFamilyHandler)

    asset_store = resolver.resolve(AssetStore)

    asset_config_overrider = resolver.resolve(AssetConfigOverrider)

    gangs = resolver.resolve(Gangs)

    return StandardEvalModelBootstrapper(
        handlers, asset_store, asset_config_overrider, gangs
    )


def _StandardEvalModelPreparer(resolver: DependencyResolver) -> EvalModelPreparer:
    gangs = resolver.resolve(Gangs)

    return StandardEvalModelPreparer(gangs)


def _StandardDistributedLogConfigurer(resolver: DependencyResolver) -> DistributedLogConfigurer:
    section = get_config_section(resolver, "common", CommonSection)

    env = resolver.resolve(Environment)

    world_info = resolver.resolve(WorldInfo)

    file_system = resolver.resolve(FileSystem)

    return StandardDistributedLogConfigurer(section, env, world_info, file_system)


def _StandardModelBootstrapper(resolver: DependencyResolver) -> ModelBootstrapper:
    section = get_config_section(resolver, "model", ModelSection)

    trainer_section = get_config_section(resolver, "trainer", TrainerSection)

    handlers = DependencyProvider(resolver, ModelFamilyHandler)

    asset_store = resolver.resolve(AssetStore)

    asset_config_overrider = resolver.resolve(AssetConfigOverrider)

    checkpoint_manager = resolver.resolve(CheckpointManager)

    gangs = resolver.resolve(Gangs)

    return StandardModelBootstrapper(
        section,
        trainer_section,
        handlers,
        asset_store,
        asset_config_overrider,
        checkpoint_manager,
        gangs,
    )


def _StandardModelPreparer(resolver: DependencyResolver) -> ModelPreparer:
    section = get_config_section(resolver, "model", ModelSection)

    trainer_section = get_config_section(resolver, "trainer", TrainerSection)

    data_parallel_wrapper = resolver.resolve(DPModelWrapper)

    return StandardModelPreparer(section, trainer_section, data_parallel_wrapper)


def _StandardModelMetadataSaver(resolver: DependencyResolver) -> ModelMetadataSaver:
    metadata_dumper = resolver.resolve(ModelMetadataDumper)

    gangs = resolver.resolve(Gangs)

    output_dir = resolver.resolve(Path)

    return StandardModelMetadataSaver(metadata_dumper, output_dir, gangs)


def _StandardOutputDirectoryCreator(resolver: DependencyResolver) -> OutputDirectoryCreator:
    sweep_tag_generator = resolver.resolve(SweepTagGenerator)

    file_system = resolver.resolve(FileSystem)

    return StandardOutputDirectoryCreator(sweep_tag_generator, file_system)


def _StandardSweepTagGenerator(resolver: DependencyResolver) -> SweepTagGenerator:
    section = get_config_section(resolver, "common", CommonSection)

    world_info = resolver.resolve(WorldInfo)

    config = resolver.resolve(object, key="config")

    value_converter = resolver.resolve(ValueConverter)

    return StandardSweepTagGenerator(section, world_info, config, value_converter)


def _StandardTorchConfigurer(resolver: DependencyResolver) -> TorchConfigurer:
    section = get_config_section(resolver, "common", CommonSection)

    world_info = resolver.resolve(WorldInfo)

    env = resolver.resolve(Environment)

    device = resolver.resolve(Device)

    rng_bag = resolver.resolve(RngBag)

    output_dir = resolver.resolve(Path)

    file_system = resolver.resolve(FileSystem)

    return StandardTorchConfigurer(
        section, world_info, env, device, rng_bag, output_dir, file_system
    )


def _StandardWandbRunIdManager(resolver: DependencyResolver) -> WandbRunIdManager:
    section = get_config_section(resolver, "common", CommonSection)

    file_system = resolver.resolve(FileSystem)

    id_generator = resolver.resolve(WandbIdGenerator)

    output_dir = resolver.resolve(Path)

    return StandardWandbRunIdManager(section, file_system, id_generator, output_dir)


def _TaskRunner(resolver: DependencyResolver) -> TaskRunner:
    task = resolver.resolve(Task)

    gangs = resolver.resolve(Gangs)

    wall_watch = resolver.resolve(Stopwatch)

    return TaskRunner(task, gangs, wall_watch)


def _TensorBoardRecorder(resolver: DependencyResolver) -> TensorBoardRecorder:
    output_dir = resolver.resolve(Path)

    metric_descriptors = DependencyProvider(resolver, MetricDescriptor)

    return TensorBoardRecorder(output_dir, metric_descriptors)


def _MaybeTensorBoardRecorderFactory(resolver: DependencyResolver) -> MaybeTensorBoardRecorderFactory:
    section = get_config_section(resolver, "common", CommonSection)

    provider = lambda: resolver.resolve(TensorBoardRecorder)

    return MaybeTensorBoardRecorderFactory(section, provider)


def _TokenizerFactory(resolver: DependencyResolver) -> TokenizerFactory:
    handlers = DependencyProvider(resolver, TokenizerFamilyHandler)

    asset_store = resolver.resolve(AssetStore)

    asset_config_overrider = resolver.resolve(AssetConfigOverrider)

    gangs = resolver.resolve(Gangs)

    return TokenizerFactory(handlers, asset_store, asset_config_overrider, gangs)


def _MaybeTorchProfilerFactory(resolver: DependencyResolver) -> MaybeTorchProfilerFactory:
    section = get_config_section(resolver, "common", CommonSection)

    output_dir = resolver.resolve(Path)

    gangs = resolver.resolve(Gangs)

    return MaybeTorchProfilerFactory(section, output_dir, gangs)


def _TriStageLRFactory(resolver: DependencyResolver) -> TriStageLRFactory:
    optimizer = resolver.resolve(Optimizer)

    regime_section = get_config_section(resolver, "regime", RegimeSection)

    return TriStageLRFactory(optimizer, regime_section)


def _WandbClientFactory(resolver: DependencyResolver) -> WandbClientFactory:
    section = get_config_section(resolver, "common", CommonSection)

    output_dir = resolver.resolve(Path)

    config = resolver.resolve(object, key="config")

    value_converter = resolver.resolve(ValueConverter)

    run_id_manager = resolver.resolve(WandbRunIdManager)

    run_factory = resolver.resolve(WandbRunFactory)

    return WandbClientFactory(
        section, output_dir, config, value_converter, run_id_manager, run_factory
    )


def _WandbRecorder(resolver: DependencyResolver) -> WandbRecorder:
    client = resolver.resolve(WandbClient)

    metric_descriptors = DependencyProvider(resolver, MetricDescriptor)

    return WandbRecorder(client, metric_descriptors)


def _MaybeWandbRecorderFactory(resolver: DependencyResolver) -> MaybeWandbRecorderFactory:
    section = get_config_section(resolver, "common", CommonSection)

    provider = lambda: resolver.resolve(WandbRecorder)

    return MaybeWandbRecorderFactory(section, provider)


def _WorldPreparer(resolver: DependencyResolver) -> WorldPreparer:
    section = get_config_section(resolver, "common", CommonSection)

    env = resolver.resolve(Environment)

    cluster_resolver = resolver.resolve(ClusterResolver)

    return WorldPreparer(section, env, cluster_resolver)
