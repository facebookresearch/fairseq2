# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.assets import (
    AssetDirectories,
    AssetMetadataLoadError,
    FileAssetMetadataLoader,
    PackageAssetMetadataLoader,
    StandardAssetMetadataFileLoader,
    WheelPackageFileLister,
)
from fairseq2.context import RuntimeContext
from fairseq2.setup._error import SetupError
from fairseq2.utils.yaml import StandardYamlLoader


def register_assets(context: RuntimeContext) -> None:
    register_package_metadata_provider(context, "fairseq2.assets.cards")

    asset_store = context.asset_store

    file_system = context.file_system

    yaml_loader = StandardYamlLoader(file_system)

    asset_metadata_file_loader = StandardAssetMetadataFileLoader(yaml_loader)

    asset_dirs = AssetDirectories(context.env, file_system)

    # /etc/fairseq2/assets
    try:
        config_dir = asset_dirs.get_system_dir()
    except AssetMetadataLoadError as ex:
        raise SetupError(
            "The system asset metadata directory cannot be determined. See the nested exception for details."
        ) from ex

    if config_dir is not None:
        metadata_loader = FileAssetMetadataLoader(
            config_dir, file_system, asset_metadata_file_loader
        )

        try:
            metadata_provider = metadata_loader.load()
        except FileNotFoundError:
            metadata_provider = None
        except AssetMetadataLoadError as ex:
            raise SetupError(
                f"The asset metadata at the '{config_dir}' path cannot be loaded. See the nested exception for details."
            ) from ex

        if metadata_provider is not None:
            asset_store.metadata_providers.append(metadata_provider)

    # ~/.config/fairseq2/assets
    try:
        config_dir = asset_dirs.get_user_dir()
    except AssetMetadataLoadError as ex:
        raise SetupError(
            "The user asset metadata directory cannot be determined. See the nested exception for details."
        ) from ex

    if config_dir is not None:
        metadata_loader = FileAssetMetadataLoader(
            config_dir, file_system, asset_metadata_file_loader
        )

        try:
            metadata_provider = metadata_loader.load()
        except FileNotFoundError:
            metadata_provider = None
        except AssetMetadataLoadError as ex:
            raise SetupError(
                f"The asset metadata at the '{config_dir}' path cannot be loaded. See the nested exception for details."
            ) from ex

        if metadata_provider is not None:
            asset_store.user_metadata_providers.append(metadata_provider)


def register_package_metadata_provider(
    context: RuntimeContext, package_name: str
) -> None:
    file_system = context.file_system

    file_lister = WheelPackageFileLister()

    yaml_loader = StandardYamlLoader(file_system)

    asset_metadata_file_loader = StandardAssetMetadataFileLoader(yaml_loader)

    metadata_loader = PackageAssetMetadataLoader(
        package_name, file_lister, asset_metadata_file_loader
    )

    try:
        provider = metadata_loader.load()
    except AssetMetadataLoadError as ex:
        raise SetupError(
            f"The asset metadata of the '{package_name}' package cannot be loaded. See the nested exception for details."
        ) from ex

    context.asset_store.metadata_providers.append(provider)
