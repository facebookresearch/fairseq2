# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fairseq2.assets import AssetMetadataProvider
from fairseq2.checkpoint import (
    CheckpointManager,
    FileCheckpointManager,
    FileCheckpointMetadataProvider,
)
from fairseq2.dependency import DependencyContainer, DependencyResolver
from fairseq2.gang import Gang
from fairseq2.recipes.config_manager import ConfigManager, ConfigNotFoundError
from fairseq2.utils.file import TensorDumper, TensorLoader
from fairseq2.utils.structured import ValueConverter


@dataclass
class FileCheckpointManagerConfig:
    path: Path = field(default_factory=lambda: Path("checkpoints"))


def register_checkpoint_manager(container: DependencyContainer) -> None:
    container.register_factory(CheckpointManager, _create_checkpoint_manager)

    container.register_factory(
        CheckpointManager, _create_file_checkpoint_manager, key="file"
    )


def _create_checkpoint_manager(resolver: DependencyResolver) -> CheckpointManager:
    config_manager = resolver.resolve(ConfigManager)

    try:
        type_ = config_manager.get_config("checkpoint_manager_type", str)
    except ConfigNotFoundError:
        type_ = "file"

    return resolver.resolve(CheckpointManager, key=type_)


def _create_file_checkpoint_manager(resolver: DependencyResolver) -> CheckpointManager:
    config_manager = resolver.resolve(ConfigManager)

    output_dir = config_manager.get_config("output_dir", Path)

    try:
        config = config_manager.get_config(
            "checkpoint_manager", FileCheckpointManagerConfig
        )
    except ConfigNotFoundError:
        config = FileCheckpointManagerConfig()

    checkpoint_dir = output_dir.joinpath(config.path)

    gang = resolver.resolve(Gang)

    dp_gang = resolver.resolve(Gang, key="dp")
    tp_gang = resolver.resolve(Gang, key="tp")

    tensor_loader = resolver.resolve(TensorLoader)
    tensor_dumper = resolver.resolve(TensorDumper)

    value_converter = resolver.resolve(ValueConverter)

    try:
        lower_score_better = config_manager.get_config("lower_score_better", bool)
    except ConfigNotFoundError:
        lower_score_better = False

    return FileCheckpointManager(
        checkpoint_dir,
        gang,
        dp_gang=dp_gang,
        tp_gang=tp_gang,
        tensor_loader=tensor_loader,
        tensor_dumper=tensor_dumper,
        value_converter=value_converter,
        lower_score_better=lower_score_better,
    )


def register_checkpoint_metadata_provider(container: DependencyContainer) -> None:
    container.register_factory(
        AssetMetadataProvider, _create_file_checkpoint_metadata_provider
    )


def _create_file_checkpoint_metadata_provider(
    resolver: DependencyResolver,
) -> AssetMetadataProvider | None:
    config_manager = resolver.resolve(ConfigManager)

    try:
        checkpoint_search_dir = config_manager.get_config(
            "checkpoint_search_dir", Path | None
        )
    except ConfigNotFoundError:
        checkpoint_search_dir = None

    if checkpoint_search_dir is None:
        return None

    try:
        lower_score_better = config_manager.get_config("lower_score_better", bool)
    except ConfigNotFoundError:
        lower_score_better = False

    return FileCheckpointMetadataProvider(
        checkpoint_search_dir, lower_score_better=lower_score_better
    )
