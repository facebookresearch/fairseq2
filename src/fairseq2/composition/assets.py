# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

from fairseq2.assets import (
    AssetDirectoryAccessor,
    AssetDownloadManager,
    AssetEnvironmentDetector,
    AssetMetadataFileLoader,
    AssetMetadataProvider,
    AssetMetadataSource,
    AssetStore,
    DelegatingAssetDownloadManager,
    FileAssetMetadataLoader,
    FileAssetMetadataSource,
    HuggingFaceHub,
    InMemoryAssetMetadataSource,
    LocalAssetDownloadManager,
    PackageAssetMetadataLoader,
    PackageAssetMetadataSource,
    PackageFileLister,
    StandardAssetDirectoryAccessor,
    StandardAssetDownloadManager,
    StandardAssetStore,
    StandardFileAssetMetadataLoader,
    StandardPackageAssetMetadataLoader,
    StandardPackageFileLister,
    WellKnownAssetMetadataSource,
    YamlAssetMetadataFileLoader,
)
from fairseq2.checkpoint import ModelMetadataSource
from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyResolver,
    wire_object,
)
from fairseq2.utils.progress import ProgressReporter


def register_file_assets(
    container: DependencyContainer, path: Path, *, not_exist_ok: bool = False
) -> None:
    def create_source(resolver: DependencyResolver) -> AssetMetadataSource:
        return wire_object(
            resolver, FileAssetMetadataSource, path=path, not_exist_ok=not_exist_ok
        )

    container.collection.register(AssetMetadataSource, create_source)


def register_package_assets(container: DependencyContainer, package: str) -> None:
    def create_source(resolver: DependencyResolver) -> AssetMetadataSource:
        return wire_object(resolver, PackageAssetMetadataSource, package=package)

    container.collection.register(AssetMetadataSource, create_source)


def register_in_memory_assets(
    container: DependencyContainer, source: str, entries: Sequence[dict[str, object]]
) -> None:
    def create_source(resolver: DependencyResolver) -> AssetMetadataSource:
        return wire_object(
            resolver, InMemoryAssetMetadataSource, name=source, entries=entries
        )

    container.collection.register(AssetMetadataSource, create_source)


def register_checkpoint_models(
    container: DependencyContainer, checkpoint_dir: Path
) -> None:
    def create_source(resolver: DependencyResolver) -> AssetMetadataSource:
        return wire_object(resolver, ModelMetadataSource, checkpoint_dir=checkpoint_dir)

    container.collection.register(AssetMetadataSource, create_source)


def _register_asset(container: DependencyContainer) -> None:
    container.register_type(AssetDirectoryAccessor, StandardAssetDirectoryAccessor)

    # Store
    def create_asset_store(resolver: DependencyResolver) -> AssetStore:
        sources = resolver.collection.resolve(AssetMetadataSource)

        def load_providers() -> Iterator[AssetMetadataProvider]:
            for source in sources:
                yield from source.load()

        env_detector = resolver.resolve(AssetEnvironmentDetector)

        env = env_detector.detect()

        return wire_object(
            resolver,
            StandardAssetStore,
            metadata_providers=load_providers(),
            default_env=env,
        )

    container.register(AssetStore, create_asset_store, singleton=True)

    container.register_type(AssetEnvironmentDetector)

    # Asset Metadata
    register_package_assets(container, package="fairseq2.assets.cards")

    container.collection.register_type(
        AssetMetadataSource, WellKnownAssetMetadataSource
    )

    container.register_type(AssetMetadataFileLoader, YamlAssetMetadataFileLoader)
    container.register_type(FileAssetMetadataLoader, StandardFileAssetMetadataLoader)
    container.register_type(PackageAssetMetadataLoader, StandardPackageAssetMetadataLoader)  # fmt: skip
    container.register_type(PackageFileLister, StandardPackageFileLister)

    # Download Manager
    container.register_type(
        AssetDownloadManager, DelegatingAssetDownloadManager, singleton=True
    )

    container.collection.register_type(AssetDownloadManager, LocalAssetDownloadManager)
    container.collection.register_type(AssetDownloadManager, HuggingFaceHub)

    def create_standard_asset_download_manager(
        resolver: DependencyResolver,
    ) -> AssetDownloadManager:
        dirs = resolver.resolve(AssetDirectoryAccessor)

        cache_dir = dirs.get_cache_dir()

        progress_reporter = resolver.resolve(ProgressReporter, key="download_reporter")

        return wire_object(
            resolver,
            StandardAssetDownloadManager,
            cache_dir=cache_dir,
            progress_reporter=progress_reporter,
        )

    container.collection.register(
        AssetDownloadManager, create_standard_asset_download_manager
    )
