# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

from fairseq2.assets import (
    AssetDownloadManager,
    AssetMetadataProvider,
    AssetMetadataSource,
    AssetStore,
    FileAssetMetadataLoader,
    _AssetDirectoryAccessor,
    _AssetEnvironmentDetector,
    _AssetMetadataFileLoader,
    _DelegatingAssetDownloadManager,
    _FileAssetMetadataSource,
    _HuggingFaceHub,
    _InMemoryAssetMetadataSource,
    _LocalAssetDownloadManager,
    _PackageAssetMetadataLoader,
    _PackageAssetMetadataSource,
    _PackageFileLister,
    _StandardAssetDirectoryAccessor,
    _StandardAssetDownloadManager,
    _StandardAssetStore,
    _StandardFileAssetMetadataLoader,
    _StandardPackageAssetMetadataLoader,
    _StandardPackageFileLister,
    _WellKnownAssetMetadataSource,
    _YamlAssetMetadataFileLoader,
)
from fairseq2.checkpoint import _ModelMetadataSource
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
            resolver, _FileAssetMetadataSource, path=path, not_exist_ok=not_exist_ok
        )

    container.collection.register(AssetMetadataSource, create_source)


def register_package_assets(container: DependencyContainer, package: str) -> None:
    def create_source(resolver: DependencyResolver) -> AssetMetadataSource:
        return wire_object(resolver, _PackageAssetMetadataSource, package=package)

    container.collection.register(AssetMetadataSource, create_source)


def register_in_memory_assets(
    container: DependencyContainer, source: str, entries: Sequence[dict[str, object]]
) -> None:
    def create_source(resolver: DependencyResolver) -> AssetMetadataSource:
        return wire_object(
            resolver, _InMemoryAssetMetadataSource, name=source, entries=entries
        )

    container.collection.register(AssetMetadataSource, create_source)


def register_checkpoint_models(
    container: DependencyContainer, checkpoint_dir: Path
) -> None:
    def create_source(resolver: DependencyResolver) -> AssetMetadataSource:
        return wire_object(
            resolver, _ModelMetadataSource, checkpoint_dir=checkpoint_dir
        )

    container.collection.register(AssetMetadataSource, create_source)


def _register_asset(container: DependencyContainer) -> None:
    container.register_type(_AssetDirectoryAccessor, _StandardAssetDirectoryAccessor)

    # Store
    def load_asset_store(resolver: DependencyResolver) -> AssetStore:
        sources = resolver.collection.resolve(AssetMetadataSource)

        def load_providers() -> Iterator[AssetMetadataProvider]:
            for source in sources:
                yield from source.load()

        metadata_providers = load_providers()

        env_detector = resolver.resolve(_AssetEnvironmentDetector)

        env = env_detector.detect()

        return wire_object(
            resolver,
            _StandardAssetStore,
            metadata_providers=metadata_providers,
            default_env=env,
        )

    container.register(AssetStore, load_asset_store, singleton=True)

    container.register_type(_AssetEnvironmentDetector)

    # Asset Metadata
    register_package_assets(container, package="fairseq2.assets.cards")

    container.collection.register_type(
        AssetMetadataSource, _WellKnownAssetMetadataSource
    )

    container.register_type(_AssetMetadataFileLoader, _YamlAssetMetadataFileLoader)
    container.register_type(FileAssetMetadataLoader, _StandardFileAssetMetadataLoader)
    container.register_type(_PackageAssetMetadataLoader, _StandardPackageAssetMetadataLoader)  # fmt: skip
    container.register_type(_PackageFileLister, _StandardPackageFileLister)

    # DownloadManager
    container.register_type(
        AssetDownloadManager, _DelegatingAssetDownloadManager, singleton=True
    )

    container.collection.register_type(AssetDownloadManager, _LocalAssetDownloadManager)
    container.collection.register_type(AssetDownloadManager, _HuggingFaceHub)

    def create_standard_asset_download_manager(
        resolver: DependencyResolver,
    ) -> AssetDownloadManager:
        dirs = resolver.resolve(_AssetDirectoryAccessor)

        cache_dir = dirs.get_cache_dir()

        progress_reporter = resolver.resolve(ProgressReporter)

        download_progress_reporter = resolver.resolve(
            ProgressReporter, key="download_reporter"
        )

        return wire_object(
            resolver,
            _StandardAssetDownloadManager,
            cache_dir=cache_dir,
            progress_reporter=progress_reporter,
            download_progress_reporter=download_progress_reporter,
        )

    container.collection.register(
        AssetDownloadManager, create_standard_asset_download_manager
    )
