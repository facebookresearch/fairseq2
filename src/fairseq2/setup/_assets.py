# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.assets import (
    FileAssetMetadataProvider,
    PackageAssetMetadataProvider,
    StandardMetadataFileLoader,
    WheelPackageFileLister,
    get_asset_dir,
    get_user_asset_dir,
)
from fairseq2.context import RuntimeContext
from fairseq2.utils.yaml import StandardYamlLoader


def _register_assets(context: RuntimeContext) -> None:
    register_package_metadata_provider(context, "fairseq2.assets.cards")

    asset_store = context.asset_store

    file_system = context.file_system

    yaml_loader = StandardYamlLoader(file_system)

    metadata_file_loader = StandardMetadataFileLoader(yaml_loader)

    # /etc/fairseq2/assets
    config_dir = get_asset_dir()
    if config_dir is not None:
        provider = FileAssetMetadataProvider(
            config_dir, file_system, metadata_file_loader
        )

        asset_store.metadata_providers.append(provider)

    # ~/.config/fairseq2/assets
    config_dir = get_user_asset_dir()
    if config_dir is not None:
        provider = FileAssetMetadataProvider(
            config_dir, file_system, metadata_file_loader
        )

        asset_store.user_metadata_providers.append(provider)


def register_package_metadata_provider(
    context: RuntimeContext, package_name: str
) -> None:
    file_lister = WheelPackageFileLister()

    yaml_loader = StandardYamlLoader(context.file_system)

    metadata_file_loader = StandardMetadataFileLoader(yaml_loader)

    provider = PackageAssetMetadataProvider(
        package_name, file_lister, metadata_file_loader
    )

    context.asset_store.metadata_providers.append(provider)
