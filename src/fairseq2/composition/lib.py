# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from rich.console import Console

from fairseq2.assets import AssetConfigLoader, StandardAssetConfigLoader
from fairseq2.checkpoint import (
    ModelMetadataDumper,
    ModelMetadataLoader,
    StandardModelMetadataDumper,
    StandardModelMetadataLoader,
)
from fairseq2.cluster import (
    ClusterHandler,
    ClusterResolver,
    SlurmHandler,
    StandardClusterResolver,
)
from fairseq2.composition.assets import _register_asset
from fairseq2.composition.datasets import _register_dataset_families
from fairseq2.composition.extensions import _register_extensions
from fairseq2.composition.models import _register_model_families
from fairseq2.composition.tokenizers import _register_tokenizer_families
from fairseq2.data.tokenizers.hub import GlobalTokenizerLoader
from fairseq2.data.tokenizers.sentencepiece import (
    SentencePieceModelLoader,
    StandardSentencePieceModelLoader,
)
from fairseq2.device import (
    CPU,
    CudaContext,
    DefaultDeviceDetector,
    Device,
    StandardCudaContext,
)
from fairseq2.file_system import FileSystem, LocalFileSystem
from fairseq2.io import (
    HuggingFaceSafetensorsLoader,
    SafetensorsLoader,
    TensorDumper,
    TensorLoader,
    TorchTensorDumper,
    TorchTensorLoader,
)
from fairseq2.model_checkpoint import (
    BasicModelCheckpointLoader,
    DelegatingModelCheckpointLoader,
    ModelCheckpointLoader,
    NativeModelCheckpointLoader,
    SafetensorsCheckpointLoader,
)
from fairseq2.models.hub import GlobalModelLoader
from fairseq2.models.llama import LLaMACheckpointLoader
from fairseq2.models.llama4.sharder import MoESharder
from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyResolver,
    wire_object,
)
from fairseq2.sharder import (
    EmbeddingSharder,
    LinearSharder,
    ModelSharder,
    ModuleSharder,
    StandardModelSharder,
)
from fairseq2.utils.config import (
    ConfigMerger,
    ConfigProcessor,
    StandardConfigMerger,
    StandardConfigProcessor,
)
from fairseq2.utils.env import Environment, StandardEnvironment
from fairseq2.utils.progress import NOOP_PROGRESS_REPORTER, ProgressReporter
from fairseq2.utils.rich import (
    RichProgressReporter,
    create_rich_download_progress_columns,
    get_error_console,
)
from fairseq2.utils.rng import RngBag
from fairseq2.utils.structured import StandardValueConverter, ValueConverter
from fairseq2.utils.threading import StandardThreadPool, ThreadPool
from fairseq2.utils.validation import ObjectValidator, StandardObjectValidator
from fairseq2.utils.yaml import (
    RuamelYamlDumper,
    RuamelYamlLoader,
    YamlDumper,
    YamlLoader,
)
from fairseq2.world_info import WorldInfo


def _register_library(
    container: DependencyContainer, *, no_progress: bool | None = None
) -> None:
    # Environment Variables
    env = StandardEnvironment()

    container.register_instance(Environment, env)

    # Console
    error_console = get_error_console()

    container.register_instance(Console, error_console)

    # Progress Reporters
    if no_progress is None:
        no_progress = env.has("FAIRSEQ2_NO_PROGRESS")

    if no_progress:
        container.register_instance(ProgressReporter, NOOP_PROGRESS_REPORTER)

        container.register_instance(
            ProgressReporter, NOOP_PROGRESS_REPORTER, key="download_reporter"
        )
    else:
        container.register_type(ProgressReporter, RichProgressReporter, singleton=True)

        def create_download_progress_reporter(
            resolver: DependencyResolver,
        ) -> ProgressReporter:
            columns = create_rich_download_progress_columns()

            return wire_object(resolver, RichProgressReporter, columns=columns)

        container.register(
            ProgressReporter,
            create_download_progress_reporter,
            key="download_reporter",
            singleton=True,
        )

    # WorldInfo
    def create_world_info(resolver: DependencyResolver) -> WorldInfo:
        env = resolver.resolve(Environment)

        return WorldInfo.from_env(env)

    container.register(WorldInfo, create_world_info, singleton=True)

    # Device
    def create_default_device(resolver: DependencyResolver) -> Device:
        device_detector = resolver.resolve(DefaultDeviceDetector)

        return device_detector.detect()

    container.register(Device, create_default_device, singleton=True)

    # ThreadPool
    def create_default_thread_pool(resolver: DependencyResolver) -> ThreadPool:
        world_info = resolver.resolve(WorldInfo)

        return StandardThreadPool.create_default(world_info.local_size)

    container.register(ThreadPool, create_default_thread_pool, singleton=True)

    # RngBag
    def create_default_rng_bag(resolver: DependencyResolver) -> RngBag:
        device = resolver.resolve(Device)

        return RngBag.from_device_defaults(CPU, device)

    container.register(RngBag, create_default_rng_bag, singleton=True)

    # fmt: off
    container.register_type(AssetConfigLoader, StandardAssetConfigLoader)
    container.register_type(ClusterResolver, StandardClusterResolver)
    container.register_type(ConfigMerger, StandardConfigMerger)
    container.register_type(ConfigProcessor, StandardConfigProcessor)
    container.register_type(CudaContext, StandardCudaContext)
    container.register_type(DefaultDeviceDetector)
    container.register_type(FileSystem, LocalFileSystem, singleton=True)
    container.register_type(GlobalModelLoader, singleton=True)
    container.register_type(GlobalTokenizerLoader, singleton=True)
    container.register_type(ModelCheckpointLoader, DelegatingModelCheckpointLoader, singleton=True)
    container.register_type(ModelMetadataDumper, StandardModelMetadataDumper)
    container.register_type(ModelMetadataLoader, StandardModelMetadataLoader)
    container.register_type(ModelSharder, StandardModelSharder, singleton=True)
    container.register_type(ObjectValidator, StandardObjectValidator, singleton=True)
    container.register_type(SafetensorsLoader, HuggingFaceSafetensorsLoader)
    container.register_type(SentencePieceModelLoader, StandardSentencePieceModelLoader, singleton=True)
    container.register_type(TensorDumper, TorchTensorDumper, singleton=True)
    container.register_type(TensorLoader, TorchTensorLoader, singleton=True)
    container.register_type(ValueConverter, StandardValueConverter, singleton=True)
    container.register_type(YamlDumper, RuamelYamlDumper, singleton=True)
    container.register_type(YamlLoader, RuamelYamlLoader, singleton=True)

    container.collection.register_type(ClusterHandler, SlurmHandler)

    container.collection.register_type(ModuleSharder, EmbeddingSharder)
    container.collection.register_type(ModuleSharder, LinearSharder)
    container.collection.register_type(ModuleSharder, MoESharder)

    container.collection.register_type(ModelCheckpointLoader, BasicModelCheckpointLoader)
    container.collection.register_type(ModelCheckpointLoader, NativeModelCheckpointLoader)
    container.collection.register_type(ModelCheckpointLoader, SafetensorsCheckpointLoader)
    container.collection.register_type(ModelCheckpointLoader, LLaMACheckpointLoader)
    # fmt: on

    _register_asset(container)

    # Asset Families
    _register_model_families(container)
    _register_dataset_families(container)
    _register_tokenizer_families(container)

    # Extensions
    _register_extensions(container)
