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
from fairseq2.recipes.config_manager import ConfigManager, register_config
from fairseq2.utils.file import TensorDumper, TensorLoader
from fairseq2.utils.structured import ValueConverter


@dataclass
class FileCheckpointManagerConfig:
    path: Path = field(default_factory=lambda: Path("checkpoints"))


@dataclass
class ScoreConfig:
    metric: str = "loss"
    lower_better: bool = True


def register_checkpoint_manager(container: DependencyContainer) -> None:
    register_config(
        container,
        path="checkpoint_manager",
        kls=FileCheckpointManagerConfig,
        default_factory=FileCheckpointManagerConfig,
    )

    register_config(container, path="output_dir", kls=Path)

    register_config(container, path="score", kls=ScoreConfig)

    container.register_factory(CheckpointManager, _create_checkpoint_manager)

    container.register_factory(
        CheckpointManager, _create_file_checkpoint_manager, key="file"
    )


def _create_checkpoint_manager(resolver: DependencyResolver) -> CheckpointManager:
    config_manager = resolver.resolve(ConfigManager)

    type_ = config_manager.get_config(
        "checkpoint_manager_type", str, default_factory=lambda: "file"
    )

    return resolver.resolve(CheckpointManager, key=type_)


def _create_file_checkpoint_manager(resolver: DependencyResolver) -> CheckpointManager:
    output_dir = resolver.resolve(Path, key="output_dir")

    config = resolver.resolve(FileCheckpointManagerConfig, key="checkpoint_manager")

    checkpoint_dir = output_dir.joinpath(config.path)

    gang = resolver.resolve(Gang)

    dp_gang = resolver.resolve(Gang, key="dp")
    tp_gang = resolver.resolve(Gang, key="tp")

    tensor_loader = resolver.resolve(TensorLoader)
    tensor_dumper = resolver.resolve(TensorDumper)

    value_converter = resolver.resolve(ValueConverter)

    score_config = resolver.resolve_optional(ScoreConfig, key="score")

    lower_score_better = False if score_config is None else score_config.lower_better

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
    register_config(
        container, path="checkpoint_search_dir", kls=Path, type_expr=Path | None
    )

    container.register_factory(
        AssetMetadataProvider, _create_file_checkpoint_metadata_provider
    )


def _create_file_checkpoint_metadata_provider(
    resolver: DependencyResolver,
) -> AssetMetadataProvider | None:
    checkpoint_search_dir = resolver.resolve_optional(Path, key="checkpoint_search_dir")
    if checkpoint_search_dir is None:
        return None

    score_config = resolver.resolve_optional(ScoreConfig, key="score")

    lower_score_better = False if score_config is None else score_config.lower_better

    return FileCheckpointMetadataProvider(
        checkpoint_search_dir, lower_score_better=lower_score_better
    )
