# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from fairseq2.assets import (
    AssetConfigLoader,
    AssetDirectoryAccessor,
    AssetDownloadManager,
    AssetEnvironmentDetector,
    AssetEnvironmentResolver,
    AssetMetadataFileLoader,
    AssetMetadataProvider,
    AssetStore,
    DelegatingAssetDownloadManager,
    FileAssetMetadataLoader,
    HuggingFaceHub,
    MaybeSystemAssetMetadataSource,
    MaybeUserAssetMetadataSource,
    NoopAssetDownloadManager,
    PackageAssetMetadataLoader,
    PackageFileLister,
    StandardAssetConfigLoader,
    StandardAssetDirectoryAccessor,
    StandardAssetDownloadManager,
    StandardAssetStore,
    StandardFileAssetMetadataLoader,
    StandardPackageAssetMetadataLoader,
    StandardPackageFileLister,
    YamlAssetMetadataFileLoader,
)
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
from fairseq2.data.tokenizers.sentencepiece import (
    SentencePieceModelLoader,
    StandardSentencePieceModelLoader,
)
from fairseq2.device import CudaContext, Device, DeviceDetector, StandardCudaContext
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
    LLaMACheckpointLoader,
    ModelCheckpointLoader,
    NativeModelCheckpointLoader,
    SafetensorsCheckpointLoader,
)
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.sharder import (
    EmbeddingSharder,
    LinearSharder,
    ModelSharder,
    ModuleSharder,
    StandardModelSharder,
)
from fairseq2.utils.config import ConfigMerger, StandardConfigMerger
from fairseq2.utils.device_stat import CudaDeviceStatTracker, DeviceStatTracker
from fairseq2.utils.env import Environment
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.rich import RichProgressReporter
from fairseq2.utils.structured import StandardValueConverter, ValueConverter
from fairseq2.utils.validation import ObjectValidator, StandardObjectValidator
from fairseq2.utils.yaml import (
    RuamelYamlDumper,
    RuamelYamlLoader,
    YamlDumper,
    YamlLoader,
)
from fairseq2.world_info import WorldInfo

# fmt: off


def _AssetEnvironmentDetector(resolver: DependencyResolver) -> AssetEnvironmentDetector:
    env_resolvers = resolver.resolve_all(AssetEnvironmentResolver)

    return AssetEnvironmentDetector(env_resolvers, resolver)


def _BasicModelCheckpointLoader(resolver: DependencyResolver) -> ModelCheckpointLoader:
    file_system = resolver.resolve(FileSystem)

    tensor_loader = resolver.resolve(TensorLoader)

    return BasicModelCheckpointLoader(file_system, tensor_loader)


def _CudaDeviceStatTracker(resolver: DependencyResolver) -> DeviceStatTracker:
    device = resolver.resolve(Device)

    cuda_context = resolver.resolve(CudaContext)

    return CudaDeviceStatTracker(device, cuda_context)


def _DelegatingAssetDownloadManager(resolver: DependencyResolver) -> AssetDownloadManager:
    managers = resolver.resolve_all(AssetDownloadManager, key="alt")

    return DelegatingAssetDownloadManager(managers)


def _DelegatingModelCheckpointLoader(resolver: DependencyResolver) -> ModelCheckpointLoader:
    loaders = resolver.resolve_all(ModelCheckpointLoader, key="alt")

    file_system = resolver.resolve(FileSystem)

    return DelegatingModelCheckpointLoader(loaders, file_system)


def _DeviceDetector(resolver: DependencyResolver) -> DeviceDetector:
    env = resolver.resolve(Environment)

    world_info = resolver.resolve(WorldInfo)

    cuda_context = resolver.resolve(CudaContext)

    return DeviceDetector(env, world_info, cuda_context)


def _EmbeddingSharder(resolver: DependencyResolver) -> ModuleSharder:
    return EmbeddingSharder()


def _HuggingFaceHub(resolver: DependencyResolver) -> AssetDownloadManager:
    return HuggingFaceHub()


def _HuggingFaceSafetensorsLoader(resolver: DependencyResolver) -> SafetensorsLoader:
    file_system = resolver.resolve(FileSystem)

    return HuggingFaceSafetensorsLoader(file_system)


def _LinearSharder(resolver: DependencyResolver) -> ModuleSharder:
    return LinearSharder()


def _LLaMACheckpointLoader(resolver: DependencyResolver) -> ModelCheckpointLoader:
    file_system = resolver.resolve(FileSystem)

    tensor_loader = resolver.resolve(TensorLoader)

    return LLaMACheckpointLoader(file_system, tensor_loader)


def _LocalFileSystem(resolver: DependencyResolver) -> FileSystem:
    return LocalFileSystem()


def _NativeModelCheckpointLoader(resolver: DependencyResolver) -> ModelCheckpointLoader:
    file_system = resolver.resolve(FileSystem)

    tensor_loader = resolver.resolve(TensorLoader)

    return NativeModelCheckpointLoader(file_system, tensor_loader)


def _NoopAssetDownloadManager(resolver: DependencyResolver) -> AssetDownloadManager:
    return NoopAssetDownloadManager()


def _RichProgressReporter(resolver: DependencyResolver) -> ProgressReporter:
    console = resolver.resolve(Console)

    world_info = resolver.resolve(WorldInfo)

    return RichProgressReporter(console, world_info)


def _RuamelYamlDumper(resolver: DependencyResolver) -> YamlDumper:
    file_system = resolver.resolve(FileSystem)

    return RuamelYamlDumper(file_system)


def _RuamelYamlLoader(resolver: DependencyResolver) -> YamlLoader:
    file_system = resolver.resolve(FileSystem)

    return RuamelYamlLoader(file_system)


def _SafetensorsCheckpointLoader(resolver: DependencyResolver) -> ModelCheckpointLoader:
    file_system = resolver.resolve(FileSystem)

    safetensors_loader = resolver.resolve(SafetensorsLoader)

    return SafetensorsCheckpointLoader(file_system, safetensors_loader)


def _SlurmHandler(resolver: DependencyResolver) -> ClusterHandler:
    env = resolver.resolve(Environment)

    return SlurmHandler(env)


def _StandardAssetConfigLoader(resolver: DependencyResolver) -> AssetConfigLoader:
    value_converter = resolver.resolve(ValueConverter)

    config_merger = resolver.resolve(ConfigMerger)

    return StandardAssetConfigLoader(value_converter, config_merger)


def _StandardAssetDirectoryAccessor(resolver: DependencyResolver) -> AssetDirectoryAccessor:
    env = resolver.resolve(Environment)

    file_system = resolver.resolve(FileSystem)

    return StandardAssetDirectoryAccessor(env, file_system)


def _StandardAssetDownloadManager(resolver: DependencyResolver, cache_dir: Path) -> AssetDownloadManager:
    return StandardAssetDownloadManager(cache_dir)


def _StandardAssetStore(resolver: DependencyResolver, env: str | None) -> AssetStore:
    metadata_providers = resolver.resolve_all(AssetMetadataProvider)

    return StandardAssetStore(metadata_providers, env)


def _StandardClusterResolver(resolver: DependencyResolver) -> ClusterResolver:
    env = resolver.resolve(Environment)

    handlers = resolver.resolve_all(ClusterHandler)

    return StandardClusterResolver(env, handlers)


def _StandardConfigMerger(resolver: DependencyResolver) -> ConfigMerger:
    return StandardConfigMerger()


def _StandardCudaContext(resolver: DependencyResolver) -> CudaContext:
    return StandardCudaContext()


def _StandardFileAssetMetadataLoader(resolver: DependencyResolver) -> FileAssetMetadataLoader:
    file_system = resolver.resolve(FileSystem)

    metadata_file_loader = resolver.resolve(AssetMetadataFileLoader)

    return StandardFileAssetMetadataLoader(file_system, metadata_file_loader)


def _StandardModelMetadataDumper(resolver: DependencyResolver) -> ModelMetadataDumper:
    file_system = resolver.resolve(FileSystem)

    yaml_dumper = resolver.resolve(YamlDumper)

    value_converter = resolver.resolve(ValueConverter)

    return StandardModelMetadataDumper(file_system, yaml_dumper, value_converter)


def _StandardModelMetadataLoader(resolver: DependencyResolver) -> ModelMetadataLoader:
    file_system = resolver.resolve(FileSystem)

    metadata_file_loader = resolver.resolve(AssetMetadataFileLoader)

    return StandardModelMetadataLoader(file_system, metadata_file_loader)


def _StandardModelSharder(resolver: DependencyResolver) -> ModelSharder:
    sharders = resolver.resolve_all(ModuleSharder)

    return StandardModelSharder(sharders)


def _StandardObjectValidator(resolver: DependencyResolver) -> ObjectValidator:
    return StandardObjectValidator()


def _StandardPackageAssetMetadataLoader(resolver: DependencyResolver) -> PackageAssetMetadataLoader:
    file_lister = resolver.resolve(PackageFileLister)

    metadata_file_loader = resolver.resolve(AssetMetadataFileLoader)

    return StandardPackageAssetMetadataLoader(file_lister, metadata_file_loader)


def _StandardPackageFileLister(resolver: DependencyResolver) -> PackageFileLister:
    return StandardPackageFileLister()


def _StandardSentencePieceModelLoader(resolver: DependencyResolver) -> SentencePieceModelLoader:
    return StandardSentencePieceModelLoader()


def _StandardValueConverter(resolver: DependencyResolver) -> ValueConverter:
    return StandardValueConverter()


def _MaybeSystemAssetMetadataSource(resolver: DependencyResolver) -> MaybeSystemAssetMetadataSource:
    dirs = resolver.resolve(AssetDirectoryAccessor)

    metadata_loader = resolver.resolve(FileAssetMetadataLoader)

    return MaybeSystemAssetMetadataSource(dirs, metadata_loader)


def _TorchTensorDumper(resolver: DependencyResolver) -> TensorDumper:
    file_system = resolver.resolve(FileSystem)

    return TorchTensorDumper(file_system)


def _TorchTensorLoader(resolver: DependencyResolver) -> TensorLoader:
    file_system = resolver.resolve(FileSystem)

    return TorchTensorLoader(file_system)


def _MaybeUserAssetMetadataSource(resolver: DependencyResolver) -> MaybeUserAssetMetadataSource:
    dirs = resolver.resolve(AssetDirectoryAccessor)

    metadata_loader = resolver.resolve(FileAssetMetadataLoader)

    return MaybeUserAssetMetadataSource(dirs, metadata_loader)


def _YamlAssetMetadataFileLoader(resolver: DependencyResolver) -> AssetMetadataFileLoader:
    yaml_loader = resolver.resolve(YamlLoader)

    return YamlAssetMetadataFileLoader(yaml_loader)
