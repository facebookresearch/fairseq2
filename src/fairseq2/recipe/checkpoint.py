# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.checkpoint import (
    CheckpointError,
    CheckpointManager,
    FileCheckpointManager,
)
from fairseq2.dependency import DependencyResolver
from fairseq2.error import SetupError
from fairseq2.file_system import FileSystem
from fairseq2.gang import Gangs
from fairseq2.recipe.config import get_recipe_output_dir
from fairseq2.utils.io import TorchTensorDumper, TorchTensorLoader
from fairseq2.utils.threading import get_default_thread_pool


def create_checkpoint_manager(resolver: DependencyResolver) -> CheckpointManager:
    gangs = resolver.resolve(Gangs)

    output_dir = get_recipe_output_dir(resolver)

    checkpoint_dir = output_dir.joinpath("checkpoints")

    file_system = resolver.resolve(FileSystem)

    tensor_loader = TorchTensorLoader(file_system)
    tensor_dumper = TorchTensorDumper(file_system)

    # TODO: move to resolver
    thread_pool = get_default_thread_pool()

    return FileCheckpointManager(
        checkpoint_dir, gangs, file_system, tensor_loader, tensor_dumper, thread_pool
    )


def check_has_checkpoint(resolver: DependencyResolver) -> bool:
    checkpoint_manager = resolver.resolve(CheckpointManager)

    try:
        return checkpoint_manager.has_checkpoint(exclude_model_only=True)
    except CheckpointError:
        raise SetupError(
            "The last training checkpoint cannot be retrieved. See the nested exception for details."  # fmt: skip
        )
