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
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess
from typing import Final, Protocol, final

from typing_extensions import override

from fairseq2.file_system import FileMode, FileSystem
from fairseq2.gang import Gangs, broadcast_flag
from fairseq2.logging import log
from fairseq2.utils.process import ProcessRunner
from fairseq2.utils.threading import ThreadPool


@dataclass(kw_only=True)
class HuggingFaceExportOptions:
    export_callback: HuggingFaceExportCallback | None = None
    blocking: bool = False


class HuggingFaceExporter(ABC):
    @abstractmethod
    def export(
        self, step_nr: int, options: HuggingFaceExportOptions | None = None
    ) -> None: ...

    @abstractmethod
    def maybe_complete_operation(self, *, blocking: bool = False) -> bool | None: ...

    @property
    @abstractmethod
    def is_exporting(self) -> bool: ...


@dataclass
class HuggingFaceExportCallbackArgs:
    step_nr: int


class HuggingFaceExportCallback(Protocol):
    def __call__(self, args: HuggingFaceExportCallbackArgs) -> None: ...


@final
class _NoopHuggingFaceExporter(HuggingFaceExporter):
    @override
    def export(
        self, step_nr: int, options: HuggingFaceExportOptions | None = None
    ) -> None:
        pass

    @override
    def maybe_complete_operation(self, *, blocking: bool = False) -> bool | None:
        return None

    @property
    @override
    def is_exporting(self) -> bool:
        return False


NOOP_HG_EXPORTER: Final = _NoopHuggingFaceExporter()


@final
class OutOfProcHuggingFaceExporter(HuggingFaceExporter):
    def __init__(
        self,
        output_dir: Path,
        gangs: Gangs,
        file_system: FileSystem,
        process_runner: ProcessRunner,
        thread_pool: ThreadPool,
    ) -> None:
        checkpoint_dir = output_dir.joinpath("checkpoints")

        self._checkpoint_dir = checkpoint_dir
        self._gangs = gangs
        self._file_system = file_system
        self._process_runner = process_runner
        self._thread_pool = thread_pool
        self._export_op: Future[Callable[[], None]] | None = None

    @override
    def export(
        self, step_nr: int, options: HuggingFaceExportOptions | None = None
    ) -> None:
        if options is None:
            options = HuggingFaceExportOptions()

        self.maybe_complete_operation(blocking=True)

        def do_export() -> Callable[[], None]:
            export_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}/hg")

            cmd = [sys.executable, "-m", "fairseq2.models.utils.hg_export", "--no-rich", "--checkpoint-dir", str(self._checkpoint_dir), f"checkpoint_step_{step_nr}", str(export_dir)]  # fmt: skip

            if self._gangs.root.rank == 0:
                run_file = export_dir.with_suffix(".run")

                fp = self._file_system.open_text(run_file, mode=FileMode.WRITE)
                with fp:
                    command_line = shlex.join(cmd)

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
                        cmd, stdout=out_fp, stderr=err_fp, env={}
                    )
            else:
                result = CompletedProcess(cmd, returncode=0)

            def commit() -> None:
                if result.returncode != 0:
                    log.warning("Hugging Face export operation of step {} failed. See operation output at {}.", step_nr, err_file)  # fmt: skip

                    return

                if options.export_callback is not None:
                    args = HuggingFaceExportCallbackArgs(step_nr)

                    options.export_callback(args)

            return commit

        if options.blocking:
            committer = do_export()

            committer()
        else:
            self._export_op = self._thread_pool.queue(do_export)

    @override
    def maybe_complete_operation(self, *, blocking: bool = False) -> bool | None:
        if self._export_op is None:
            return None

        gangs = self._gangs

        if blocking:
            committer = self._export_op.result()

            gangs.root.barrier()
        else:
            if gangs.root.rank == 0:
                done = self._export_op.done()
            else:
                done = True

            done = broadcast_flag(gangs.root, done)

            if not done:
                return False

            committer = self._export_op.result()

        self._export_op = None

        committer()

        return True

    @property
    @override
    def is_exporting(self) -> bool:
        return self._export_op is not None
