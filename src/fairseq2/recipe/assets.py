# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.assets import (
    AssetMetadataProvider,
    FileBackedAssetMetadataLoader,
    YamlAssetMetadataFileLoader,
)
from fairseq2.checkpoint import CheckpointAssetMetadataLoader
from fairseq2.error import InfraError
from fairseq2.file_system import FileSystem
from fairseq2.logging import log
from fairseq2.recipe.config import CommonSection, get_config_section
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.utils.yaml import YamlLoader


def _maybe_load_extra_assets(
    resolver: DependencyResolver,
) -> AssetMetadataProvider | None:
    common_section = get_config_section(resolver, "common", CommonSection)

    extra_path = common_section.assets.extra_path
    if extra_path is None:
        return None

    file_system = resolver.resolve(FileSystem)

    yaml_loader = resolver.resolve(YamlLoader)

    metadata_file_loader = YamlAssetMetadataFileLoader(yaml_loader)

    metadata_loader = FileBackedAssetMetadataLoader(file_system, metadata_file_loader)

    try:
        extra_path = file_system.resolve(extra_path)
    except OSError as ex:
        raise InfraError(
            f"A system error has occurred while resolving the '{extra_path}' extra asset path. See the nested exception for details."
        ) from ex

    try:
        return metadata_loader.load(extra_path)
    except FileNotFoundError:
        log.warning("'{}' path pointed to by `common.assets.extra_path` does not exist.", extra_path)  # fmt: skip

    return None


def _maybe_load_checkpoint_assets(
    resolver: DependencyResolver,
) -> AssetMetadataProvider | None:
    common_section = get_config_section(resolver, "common", CommonSection)

    checkpoint_dir = common_section.assets.checkpoint_dir
    if checkpoint_dir is None:
        return None

    file_system = resolver.resolve(FileSystem)

    yaml_loader = resolver.resolve(YamlLoader)

    metadata_file_loader = YamlAssetMetadataFileLoader(yaml_loader)

    metadata_loader = CheckpointAssetMetadataLoader(file_system, metadata_file_loader)

    try:
        checkpoint_dir = file_system.resolve(checkpoint_dir)
    except OSError as ex:
        raise InfraError(
            f"A system error has occurred while resolving the '{checkpoint_dir}' checkpoint directory. See the nested exception for details."
        ) from ex

    try:
        return metadata_loader.load(checkpoint_dir)
    except FileNotFoundError:
        log.warning("The checkpoint metadata file (model.yaml) is not found under '{}'. Make sure that `common.assets.checkpoint_dir` points to the checkpoint directory used during training.", checkpoint_dir)  # fmt: skip

    return None
