# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.assets import (
    FileAssetMetadataProvider,
    PackageAssetMetadataProvider,
    StandardAssetStore,
    WheelPackageFileLister,
    get_asset_dir,
    get_user_asset_dir,
)
from fairseq2.context import RuntimeContext
from fairseq2.utils.file import StandardFileSystem
from fairseq2.utils.yaml import load_yaml


def _register_assets(context: RuntimeContext) -> None:
    asset_store = context.asset_store

    # Package Metadata
    register_package_metadata_provider(asset_store, "fairseq2.assets.cards")

    # /etc/fairseq2/assets
    _register_asset_dir(asset_store)

    # ~/.config/fairseq2/assets
    _register_user_asset_dir(asset_store)


def _register_asset_dir(asset_store: StandardAssetStore) -> None:
    config_dir = get_asset_dir()
    if config_dir is None:
        return

    file_system = StandardFileSystem()

    provider = FileAssetMetadataProvider(config_dir, file_system, load_yaml)

    asset_store.metadata_providers.append(provider)


def _register_user_asset_dir(asset_store: StandardAssetStore) -> None:
    config_dir = get_user_asset_dir()
    if config_dir is None:
        return

    file_system = StandardFileSystem()

    provider = FileAssetMetadataProvider(config_dir, file_system, load_yaml)

    asset_store.user_metadata_providers.append(provider)


def register_package_metadata_provider(
    asset_store: StandardAssetStore, package_name: str
) -> None:
    package_file_lister = WheelPackageFileLister()

    provider = PackageAssetMetadataProvider(
        package_name, package_file_lister, load_yaml
    )

    asset_store.metadata_providers.append(provider)
