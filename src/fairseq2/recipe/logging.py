# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from logging import DEBUG, FileHandler, Formatter, getLogger
from pathlib import Path
from typing import TYPE_CHECKING

from fairseq2n import DOC_MODE

from fairseq2.error import InfraError
from fairseq2.file_system import FileSystem
from fairseq2.logging import log
from fairseq2.recipe.cluster import WorldInfo
from fairseq2.recipe.config import CommonSection, get_config_section, get_output_dir
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.utils.env import get_env


def _configure_distributed_logging(resolver: DependencyResolver) -> None:
    common_section = get_config_section(resolver, "common", CommonSection)

    file_system = resolver.resolve(FileSystem)

    world_info = resolver.resolve(WorldInfo)

    output_dir = get_output_dir(resolver)

    logger = getLogger("fairseq2")

    rank = world_info.rank

    if rank != 0:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

            handler.close()

    log_file = output_dir.joinpath(f"logs/rank_{rank}.log")

    try:
        file_system.make_directory(log_file.parent)
    except OSError as ex:
        raise InfraError(
            f"A system error has occurred while creating the '{log_file}' log file. See the nested exception for details."
        ) from ex

    handler = FileHandler(log_file)

    fmt = f"[Rank {rank}] %(asctime)s %(levelname)s %(name)s - %(message)s"

    file_formatter = Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    handler.setFormatter(file_formatter)

    logger.addHandler(handler)

    if common_section.debug:
        logger.setLevel(DEBUG)

    _configure_aten_logging(resolver, log_file)

    _configure_nccl_logging(resolver, log_file)

    log.info("Log files are stored under {}.", output_dir)


def _configure_aten_logging(resolver: DependencyResolver, log_file: Path) -> None:
    file_system = resolver.resolve(FileSystem)

    env = get_env(resolver)

    if "TORCH_CPP_LOG_LEVEL" in env:
        return

    aten_log_file = log_file.parent.joinpath("aten", log_file.name)

    try:
        file_system.make_directory(aten_log_file.parent)
    except OSError as ex:
        raise InfraError(
            f"A system error has occurred while creating the '{aten_log_file.parent}' ATen log directory. See the nested exception for details."
        ) from ex

    _enable_aten_logging(aten_log_file)

    # This variable has no effect at this point; set for completeness.
    env["TORCH_CPP_LOG_LEVEL"] = "INFO"


def _configure_nccl_logging(resolver: DependencyResolver, log_file: Path) -> None:
    file_system = resolver.resolve(FileSystem)

    env = get_env(resolver)

    if "NCCL_DEBUG" in env:
        return

    nccl_log_file = log_file.parent.joinpath("nccl", log_file.name)

    try:
        file_system.make_directory(nccl_log_file.parent)
    except OSError as ex:
        raise InfraError(
            f"A system error has occurred while creating the '{nccl_log_file.parent}' NCCL log directory. See the nested exception for details."
        ) from ex

    env["NCCL_DEBUG"] = "INFO"
    env["NCCL_DEBUG_FILE"] = str(nccl_log_file)


if TYPE_CHECKING or DOC_MODE:

    def _enable_aten_logging(log_file: Path) -> None: ...

else:
    from fairseq2n.bindings import _enable_aten_logging
