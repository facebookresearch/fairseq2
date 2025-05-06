# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import atexit
from pathlib import Path

from fairseq2.checkpoint import (
    CheckpointManager,
    CheckpointSaver,
    FileCheckpointManager,
    InProcCheckpointSaver,
    OutOfProcCheckpointSaver,
)
from fairseq2.context import RuntimeContext
from fairseq2.gang import Gangs
from fairseq2.recipes.config import RegimeSection
from fairseq2.utils.io import TorchTensorDumper, TorchTensorLoader
from fairseq2.utils.threading import get_default_thread_pool


def create_checkpoint_manager(
    context: RuntimeContext,
    regime_section: RegimeSection,
    gangs: Gangs,
    output_dir: Path,
) -> CheckpointManager:
    checkpoint_dir = output_dir.joinpath("checkpoints")

    file_system = context.file_system

    tensor_loader = TorchTensorLoader(file_system)
    tensor_dumper = TorchTensorDumper(file_system)

    saver: CheckpointSaver

    if regime_section.in_proc_checkpoint:
        saver = InProcCheckpointSaver(tensor_dumper)
    else:
        saver = OutOfProcCheckpointSaver(tensor_dumper)

        atexit.register(saver.close)

    thread_pool = get_default_thread_pool()

    return FileCheckpointManager(
        checkpoint_dir, gangs, file_system, saver, tensor_loader, thread_pool
    )
