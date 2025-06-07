# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from rich.console import Console

from fairseq2.assets import (
    AssetConfigLoader,
    AssetDirectoryAccessor,
    AssetDownloadManager,
    AssetMetadataFileLoader,
    AssetMetadataProvider,
    AssetStore,
    FileAssetMetadataLoader,
    PackageAssetMetadataLoader,
    PackageFileLister,
    register_package_assets,
)
from fairseq2.checkpoint import ModelMetadataDumper, ModelMetadataLoader
from fairseq2.cluster import ClusterHandler, ClusterResolver
from fairseq2.composition.extensions import _register_extensions
from fairseq2.composition.models import _register_model_families
from fairseq2.composition.tokenizers import _register_tokenizer_families
from fairseq2.composition.wire import (
    _AssetEnvironmentDetector,
    _BasicModelCheckpointLoader,
    _CudaDeviceStatTracker,
    _DelegatingAssetDownloadManager,
    _DelegatingModelCheckpointLoader,
    _DeviceDetector,
    _EmbeddingSharder,
    _HuggingFaceHub,
    _HuggingFaceSafetensorsLoader,
    _LinearSharder,
    _LLaMACheckpointLoader,
    _LocalFileSystem,
    _MaybeSystemAssetMetadataSource,
    _MaybeUserAssetMetadataSource,
    _NativeModelCheckpointLoader,
    _NoopAssetDownloadManager,
    _RichProgressReporter,
    _RuamelYamlDumper,
    _RuamelYamlLoader,
    _SafetensorsCheckpointLoader,
    _SlurmHandler,
    _StandardAssetConfigLoader,
    _StandardAssetDirectoryAccessor,
    _StandardAssetDownloadManager,
    _StandardAssetStore,
    _StandardClusterResolver,
    _StandardConfigMerger,
    _StandardCudaContext,
    _StandardFileAssetMetadataLoader,
    _StandardModelMetadataDumper,
    _StandardModelMetadataLoader,
    _StandardModelSharder,
    _StandardObjectValidator,
    _StandardPackageAssetMetadataLoader,
    _StandardPackageFileLister,
    _StandardSentencePieceModelLoader,
    _StandardValueConverter,
    _TorchTensorDumper,
    _TorchTensorLoader,
    _YamlAssetMetadataFileLoader,
)
from fairseq2.data.tokenizers.sentencepiece import SentencePieceModelLoader
from fairseq2.device import CPU, CudaContext, Device
from fairseq2.file_system import FileSystem
from fairseq2.io import SafetensorsLoader, TensorDumper, TensorLoader
from fairseq2.model_checkpoint import ModelCheckpointLoader
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver
from fairseq2.sharder import ModelSharder, ModuleSharder
from fairseq2.utils.config import ConfigMerger
from fairseq2.utils.device_stat import DeviceStatTracker
from fairseq2.utils.env import Environment, StandardEnvironment
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.rich import get_error_console
from fairseq2.utils.rng import RngBag
from fairseq2.utils.structured import ValueConverter
from fairseq2.utils.threading import StandardThreadPool, ThreadPool
from fairseq2.utils.validation import ObjectValidator
from fairseq2.utils.yaml import YamlDumper, YamlLoader
from fairseq2.world_info import WorldInfo


def _register_library(container: DependencyContainer) -> None:
    # Environment Variables
    container.register_instance(Environment, StandardEnvironment())

    # Console
    container.register_instance(Console, get_error_console())

    # WorldInfo
    def create_world_info(resolver: DependencyResolver) -> WorldInfo:
        env = resolver.resolve(Environment)

        return WorldInfo.from_env(env)

    container.register(WorldInfo, create_world_info)

    # Device
    def create_device(resolver: DependencyResolver) -> Device:
        return _DeviceDetector(resolver).detect()

    container.register(Device, create_device)

    # ThreadPool
    def create_thread_pool(resolver: DependencyResolver) -> ThreadPool:
        world_info = resolver.resolve(WorldInfo)

        return StandardThreadPool.create_default(world_info.local_size)

    container.register(ThreadPool, create_thread_pool)

    # RngBag
    def create_rng_bag(resolver: DependencyResolver) -> RngBag:
        device = resolver.resolve(Device)

        return RngBag.from_device_defaults(CPU, device)

    container.register(RngBag, create_rng_bag)

    # AssetStore
    def create_asset_store(resolver: DependencyResolver) -> AssetStore:
        env_detector = _AssetEnvironmentDetector(resolver)

        env = env_detector.detect()

        return _StandardAssetStore(resolver, env)

    container.register(AssetStore, create_asset_store)

    # Asset Metadata
    register_package_assets(container, package="fairseq2.assets.cards")

    # Asset Metadata
    def create_system_asset_metadata_provider(
        resolver: DependencyResolver,
    ) -> AssetMetadataProvider | None:
        return _MaybeSystemAssetMetadataSource(resolver).maybe_load()

    container.register(AssetMetadataProvider, create_system_asset_metadata_provider)

    # Asset Metadata
    def create_user_asset_metadata_provider(
        resolver: DependencyResolver,
    ) -> AssetMetadataProvider | None:
        return _MaybeUserAssetMetadataSource(resolver).maybe_load()

    container.register(AssetMetadataProvider, create_user_asset_metadata_provider)

    # Wire
    container.register(AssetConfigLoader, _StandardAssetConfigLoader)
    container.register(AssetDirectoryAccessor, _StandardAssetDirectoryAccessor)
    container.register(AssetDownloadManager, _DelegatingAssetDownloadManager)
    container.register(AssetMetadataFileLoader, _YamlAssetMetadataFileLoader)
    container.register(ClusterHandler, _SlurmHandler)
    container.register(ClusterResolver, _StandardClusterResolver)
    container.register(ConfigMerger, _StandardConfigMerger)
    container.register(CudaContext, _StandardCudaContext)
    container.register(DeviceStatTracker, _CudaDeviceStatTracker, key="cuda")
    container.register(FileAssetMetadataLoader, _StandardFileAssetMetadataLoader)
    container.register(FileSystem, _LocalFileSystem)
    container.register(ModelCheckpointLoader, _DelegatingModelCheckpointLoader)
    container.register(ModelMetadataDumper, _StandardModelMetadataDumper)
    container.register(ModelMetadataLoader, _StandardModelMetadataLoader)
    container.register(ModelSharder, _StandardModelSharder)
    container.register(ModuleSharder, _EmbeddingSharder)
    container.register(ModuleSharder, _LinearSharder)
    container.register(ObjectValidator, _StandardObjectValidator)
    container.register(PackageAssetMetadataLoader, _StandardPackageAssetMetadataLoader)
    container.register(PackageFileLister, _StandardPackageFileLister)
    container.register(ProgressReporter, _RichProgressReporter)
    container.register(SafetensorsLoader, _HuggingFaceSafetensorsLoader)
    container.register(SentencePieceModelLoader, _StandardSentencePieceModelLoader)
    container.register(TensorDumper, _TorchTensorDumper)
    container.register(TensorLoader, _TorchTensorLoader)
    container.register(ValueConverter, _StandardValueConverter)
    container.register(YamlDumper, _RuamelYamlDumper)
    container.register(YamlLoader, _RuamelYamlLoader)

    # AssetDownloadManagers
    container.register(AssetDownloadManager, _NoopAssetDownloadManager, key="alt")
    container.register(AssetDownloadManager, _HuggingFaceHub, key="alt")

    def create_standard_asset_download_manager(
        resolver: DependencyResolver,
    ) -> AssetDownloadManager:
        dirs = resolver.resolve(AssetDirectoryAccessor)

        cache_dir = dirs.get_cache_dir()

        return _StandardAssetDownloadManager(resolver, cache_dir)

    container.register(
        AssetDownloadManager, create_standard_asset_download_manager, key="alt"
    )

    # ModelCheckpointLoaders
    container.register(ModelCheckpointLoader, _BasicModelCheckpointLoader, key="alt")
    container.register(ModelCheckpointLoader, _NativeModelCheckpointLoader, key="alt")
    container.register(ModelCheckpointLoader, _SafetensorsCheckpointLoader, key="alt")
    container.register(ModelCheckpointLoader, _LLaMACheckpointLoader, key="alt")

    # Asset Families
    _register_model_families(container)

    _register_tokenizer_families(container)

    _register_extensions(container)
