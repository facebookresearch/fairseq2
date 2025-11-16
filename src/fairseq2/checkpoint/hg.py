# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import shlex
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import ExitStack
from pathlib import Path
from typing import Final, final

from typing_extensions import override

from fairseq2.error import InvalidOperationError
from fairseq2.file_system import FileMode, FileSystem
from fairseq2.logging import log
from fairseq2.utils.process import ProcessRunner
from fairseq2.utils.threading import ThreadPool


class CheckpointHGExporter(ABC):
    @abstractmethod
    def export(
        self,
        step_nr: int,
        *,
        exported_callback: Callable[[int], None] | None = None,
        blocking: bool = False,
    ) -> None: ...

    @abstractmethod
    def complete_pending(self) -> None: ...

    @property
    @abstractmethod
    def is_exporting(self) -> bool: ...


@final
class _NoopCheckpointHGExporter(CheckpointHGExporter):
    @override
    def export(
        self,
        step_nr: int,
        *,
        exported_callback: Callable[[int], None] | None = None,
        blocking: bool = False,
    ) -> None:
        pass

    @override
    def complete_pending(self) -> None:
        pass

    @property
    @override
    def is_exporting(self) -> bool:
        return False


NOOP_CHECKPOINT_HG_EXPORTER: Final = _NoopCheckpointHGExporter()


@final
class OutOfProcCheckpointHGExporter(CheckpointHGExporter):
    def __init__(
        self,
        output_dir: Path,
        file_system: FileSystem,
        process_runner: ProcessRunner,
        thread_pool: ThreadPool,
    ) -> None:
        checkpoint_dir = output_dir.joinpath("checkpoints")

        self._checkpoint_dir = checkpoint_dir
        self._file_system = file_system
        self._process_runner = process_runner
        self._thread_pool = thread_pool
        self._export_op: Future[None] | None = None

    @override
    def export(
        self,
        step_nr: int,
        *,
        exported_callback: Callable[[int], None] | None = None,
        blocking: bool = False,
    ) -> None:
        if self._export_op is not None:
            if not self._export_op.done():
                raise InvalidOperationError(
                    "A Hugging Face export operation is already in progress."
                )

            self._export_op = None

        def do_export() -> None:
            export_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}/hg")

            args: list[str] = [sys.executable, "-m", "fairseq2.models.utils.hg_export", "--no-rich", "--checkpoint-dir", str(self._checkpoint_dir), f"checkpoint_step_{step_nr}", str(export_dir)]  # fmt: skip

            run_file = export_dir.with_suffix(".run")

            fp = self._file_system.open_text(run_file, mode=FileMode.WRITE)
            with fp:
                command_line = shlex.join(args)

                fp.write(command_line)

            out_file = export_dir.with_suffix(".stdout")
            err_file = export_dir.with_suffix(".stderr")

            with ExitStack() as exit_stack:
                out_fp = exit_stack.enter_context(
                    self._file_system.open_text(out_file, mode=FileMode.WRITE)
                )

                err_fp = exit_stack.enter_context(
                    self._file_system.open_text(err_file, mode=FileMode.WRITE)
                )

                result = self._process_runner.run_text(
                    args, stdout=out_fp, stderr=err_fp, env={}
                )

            if result.returncode != 0:
                log.warning("Hugging Face export operation of step {} failed. See operation output at {}.", step_nr, err_file)  # fmt: skip
            elif exported_callback is not None:
                exported_callback(step_nr)

        if blocking:
            do_export()
        else:
            self._export_op = self._thread_pool.queue(do_export)

    @override
    def complete_pending(self) -> None:
        if self._export_op is None:
            return

        self._export_op.result()

        self._export_op = None

    @property
    @override
    def is_exporting(self) -> bool:
        if self._export_op is not None:
            return not self._export_op.done()

        return False
