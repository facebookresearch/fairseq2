# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from fairseq2.assets import (
    AssetMetadataFileLoader,
    AssetMetadataLoadError,
    FileAssetMetadataLoader,
    StandardAssetMetadataFileLoader,
    StandardAssetStore,
)
from fairseq2.checkpoint import FileCheckpointMetadataLoader
from fairseq2.context import RuntimeContext
from fairseq2.file_system import FileSystem
from fairseq2.logging import log
from fairseq2.recipes import RecipeError
from fairseq2.recipes.config import AssetsSection
from fairseq2.utils.yaml import RuamelYamlLoader


def register_extra_asset_paths(
    context: RuntimeContext, assets_section: AssetsSection
) -> None:
    asset_store = context.asset_store

    file_system = context.file_system

    yaml_loader = RuamelYamlLoader(file_system)

    asset_metadata_file_loader = StandardAssetMetadataFileLoader(yaml_loader)

    extra_path_registrar = _ExtraPathRegistrar(
        asset_store, file_system, asset_metadata_file_loader
    )

    try:
        extra_path_registrar.register(assets_section)
    except AssetMetadataLoadError as ex:
        raise RecipeError(
            "`common.assets.extra_path` cannot be registered as an asset card path. See the nested exception for details."
        ) from ex

    checkpoint_dir_registrar = _CheckpointDirectoryRegistrar(
        asset_store, file_system, asset_metadata_file_loader
    )

    try:
        checkpoint_dir_registrar.register(assets_section)
    except AssetMetadataLoadError as ex:
        raise RecipeError(
            "`common.assets.checkpoint_dir` cannot be registered as an asset card path. See the nested exception for details."
        ) from ex


@final
class _ExtraPathRegistrar:
    _asset_store: StandardAssetStore
    _file_system: FileSystem
    _asset_metadata_file_loader: AssetMetadataFileLoader

    def __init__(
        self,
        asset_store: StandardAssetStore,
        file_system: FileSystem,
        asset_metadata_file_loader: AssetMetadataFileLoader,
    ) -> None:
        self._asset_store = asset_store
        self._file_system = file_system
        self._asset_metadata_file_loader = asset_metadata_file_loader

    def register(self, assets_section: AssetsSection) -> None:
        extra_path = assets_section.extra_path
        if extra_path is None:
            return

        try:
            extra_path = self._file_system.resolve(extra_path)
        except OSError as ex:
            raise AssetMetadataLoadError(
                f"The '{extra_path}' extra asset card path cannot be read. See the nested exception for details."
            ) from ex

        file_metadata_loader = FileAssetMetadataLoader(
            extra_path, self._file_system, self._asset_metadata_file_loader
        )

        try:
            metadata_provider = file_metadata_loader.load()
        except FileNotFoundError:
            log.warning("'{}' path pointed to by `common.assets.extra_path` does not exist.", extra_path)  # fmt: skip

            return

        self._asset_store.user_metadata_providers.append(metadata_provider)


@final
class _CheckpointDirectoryRegistrar:
    _asset_store: StandardAssetStore
    _file_system: FileSystem
    _asset_metadata_file_loader: AssetMetadataFileLoader

    def __init__(
        self,
        asset_store: StandardAssetStore,
        file_system: FileSystem,
        asset_metadata_file_loader: AssetMetadataFileLoader,
    ) -> None:
        self._asset_store = asset_store
        self._file_system = file_system
        self._asset_metadata_file_loader = asset_metadata_file_loader

    def register(self, assets_section: AssetsSection) -> None:
        checkpoint_dir = assets_section.checkpoint_dir
        if checkpoint_dir is None:
            return

        try:
            checkpoint_dir = self._file_system.resolve(checkpoint_dir)
        except OSError as ex:
            raise AssetMetadataLoadError(
                f"The '{checkpoint_dir}' checkpoint directory cannot be accessed. See the nested exception for details."
            ) from ex

        checkpoint_metadata_loader = FileCheckpointMetadataLoader(
            checkpoint_dir, self._file_system, self._asset_metadata_file_loader
        )

        try:
            metadata_provider = checkpoint_metadata_loader.load()
        except FileNotFoundError:
            log.warning("The checkpoint metadata file (model.yaml) is not found under '{}'. Make sure that `common.assets.checkpoint_dir` points to the base checkpoint directory used during training.", checkpoint_dir)  # fmt: skip

            return

        self._asset_store.user_metadata_providers.append(metadata_provider)
