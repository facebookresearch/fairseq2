# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import atexit
import tempfile
from logging import DEBUG, FileHandler, Formatter
from pathlib import Path
from typing import TYPE_CHECKING, final

from fairseq2n import DOC_MODE

from fairseq2.error import raise_operational_system_error
from fairseq2.file_system import FileSystem
from fairseq2.logging import log
from fairseq2.recipe.config import CommonSection
from fairseq2.utils.env import Environment
from fairseq2.world_info import WorldInfo


@final
class _DistributedLogConfigurer:
    def __init__(
        self,
        output_dir: Path,
        section: CommonSection,
        env: Environment,
        world_info: WorldInfo,
        file_system: FileSystem,
    ) -> None:
        self._output_dir = output_dir
        self._section = section
        self._env = env
        self._world_info = world_info
        self._file_system = file_system
        # For remote output dirs, logs are written locally
        self._local_log_dir: Path | None = None
        self._is_remote = False
        self._file_handler: FileHandler | None = None

    def configure(self) -> None:
        self._configure_logger()

        self._configure_aten_logging()

        self._configure_nccl_logging()

        # Register cleanup to sync logs on exit
        if self._is_remote:
            atexit.register(self._sync_logs_to_remote)

        if self._local_log_dir is not None:
            log.info(
                "Log files are stored locally under {} (output_dir is remote: {}).",
                self._local_log_dir,
                self._output_dir,
            )
        else:
            log.info("Log files are stored under {}.", self._output_dir)

    def _get_log_base_dir(self) -> Path:
        """Get the base directory for log files.

        For local output directories, returns output_dir.
        For remote output directories (S3, etc.), returns a local temp directory.
        """
        if self._file_system.is_local_path(self._output_dir):
            return self._output_dir

        # For remote filesystems, use a local directory for logs
        self._is_remote = True
        if self._local_log_dir is None:
            # Create a local log directory based on output_dir name
            output_name = str(self._output_dir).replace("://", "_").replace("/", "_")
            self._local_log_dir = (
                Path(tempfile.gettempdir()) / f"fairseq2_logs_{output_name}"
            )
            self._local_log_dir.mkdir(parents=True, exist_ok=True)

        return self._local_log_dir

    def _sync_logs_to_remote(self) -> None:
        """Sync local log files to remote storage."""
        if not self._is_remote or self._local_log_dir is None:
            return

        # Close file handler first to flush all data
        if self._file_handler is not None:
            self._file_handler.close()

        remote_logs_dir = self._output_dir / "logs"

        try:
            # Copy all log files from local to remote
            local_logs_dir = self._local_log_dir / "logs"
            if local_logs_dir.exists():
                for local_path in local_logs_dir.rglob("*"):
                    if local_path.is_file():
                        rel_path = local_path.relative_to(local_logs_dir)
                        remote_path = remote_logs_dir.joinpath(rel_path)

                        # Read local file and write to remote
                        with open(local_path, "rb") as f:
                            data = f.read()

                        self._file_system.make_directory(remote_path.parent)
                        with self._file_system.open_for_write(remote_path) as f:
                            f.write(data)

                log.info("Synced logs to remote: {}", remote_logs_dir)
        except Exception as e:
            log.warning("Failed to sync logs to remote: {}", e)

    def _configure_logger(self) -> None:
        logger = log.logger

        rank = self._world_info.rank

        if rank != 0:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

                handler.close()

        log_base = self._get_log_base_dir()
        log_file = log_base / "logs" / f"rank_{rank}.log"

        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as ex:
            raise_operational_system_error(ex)

        try:
            self._file_handler = FileHandler(log_file)
        except OSError as ex:
            raise_operational_system_error(ex)

        fmt = f"[Rank {rank}] %(asctime)s %(levelname)s %(name)s - %(message)s"

        file_formatter = Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

        self._file_handler.setFormatter(file_formatter)

        logger.addHandler(self._file_handler)

        if self._section.debug:
            logger.setLevel(DEBUG)

    def _configure_aten_logging(self) -> None:
        if self._env.has("TORCH_CPP_LOG_LEVEL"):
            return

        log_base = self._get_log_base_dir()
        log_file = log_base / "logs" / "aten" / f"rank_{self._world_info.rank}.log"

        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as ex:
            raise_operational_system_error(ex)

        _enable_aten_logging(log_file)

        # This variable has no effect at this point; set for completeness.
        self._env.set("TORCH_CPP_LOG_LEVEL", "INFO")

    def _configure_nccl_logging(self) -> None:
        if self._env.has("NCCL_DEBUG"):
            return

        log_base = self._get_log_base_dir()
        log_file = log_base / "logs" / "nccl" / f"rank_{self._world_info.rank}.log"

        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as ex:
            raise_operational_system_error(ex)

        self._env.set("NCCL_DEBUG", "INFO")
        self._env.set("NCCL_DEBUG_FILE", str(log_file))


if TYPE_CHECKING or DOC_MODE:

    def _enable_aten_logging(log_file: Path) -> None: ...

else:
    from fairseq2n.bindings import _enable_aten_logging
