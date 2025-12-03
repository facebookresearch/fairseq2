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

from fairseq2.checkpoint.manager import CheckpointError
from fairseq2.error import InternalError
from fairseq2.file_system import FileMode, FileSystem
from fairseq2.gang import GangError, Gangs, broadcast_flag
from fairseq2.utils.process import ProcessRunner
from fairseq2.utils.threading import ThreadPool


@dataclass
class HuggingFaceExportCallbackArgs:
    step_nr: int


class HuggingFaceExportCallback(Protocol):
    def __call__(self, args: HuggingFaceExportCallbackArgs) -> None: ...


@dataclass(kw_only=True)
class HuggingFaceExportOptions:
    export_callback: HuggingFaceExportCallback | None = None
    blocking: bool = False


class HuggingFaceExporter(ABC):
    @abstractmethod
    def export(
        self, step_nr: int, options: HuggingFaceExportOptions | None = None
    ) -> None:
        """
        :raises CheckpointError:
        """

    @abstractmethod
    def maybe_complete_operation(self, *, blocking: bool = False) -> bool | None:
        """
        :raises CheckpointError:
        """

    @property
    @abstractmethod
    def step_nr(self) -> int | None: ...


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
    def step_nr(self) -> int | None:
        return None


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
        self._step_nr: int | None = None

    @override
    def export(
        self, step_nr: int, options: HuggingFaceExportOptions | None = None
    ) -> None:
        if options is None:
            options = HuggingFaceExportOptions()

        self.maybe_complete_operation(blocking=True)

        def do_export() -> Callable[[], None]:
            export_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}/hg")

            out_file = export_dir.with_suffix(".stdout")
            err_file = export_dir.with_suffix(".stderr")

            cmd = [sys.executable, "-m", "fairseq2.models.utils.hg_export", "--no-rich", "--checkpoint-dir", str(self._checkpoint_dir), f"checkpoint_step_{step_nr}", str(export_dir)]  # fmt: skip

            if self._gangs.root.rank == 0:
                command_line = shlex.join(cmd)

                run_file = export_dir.with_suffix(".run")

                try:
                    fp = self._file_system.open_text(run_file, mode=FileMode.WRITE)
                    with fp:
                        fp.write(command_line)
                except OSError as ex:
                    raise CheckpointError(f"failed to write file '{run_file}'") from ex

                with ExitStack() as exit_stack:
                    try:
                        out_fp = exit_stack.enter_context(
                            self._file_system.open_text(out_file, mode=FileMode.WRITE)
                        )
                    except OSError as ex:
                        raise CheckpointError(
                            f"failed to open file '{out_file}'"
                        ) from ex

                    try:
                        err_fp = exit_stack.enter_context(
                            self._file_system.open_text(err_file, mode=FileMode.WRITE)
                        )
                    except OSError as ex:
                        raise CheckpointError(
                            f"failed to open file '{err_file}'"
                        ) from ex

                    result = self._process_runner.run_text(
                        cmd, stdout=out_fp, stderr=err_fp, env={}
                    )
            else:
                result = CompletedProcess(cmd, returncode=0)

            def commit() -> None:
                if result.returncode != 0:
                    raise CheckpointError(
                        f"failed to export Hugging Face model of step {step_nr}, see operation output at {err_file}"
                    )

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
        step_nr = self._step_nr

        if step_nr is None:
            return None

        if self._export_op is None:
            raise InternalError(f"``step_nr` is {step_nr}, but `export_op` is `None`.")

        gangs = self._gangs

        if blocking:
            committer = self._export_op.result()

            try:
                gangs.root.barrier()
            except GangError as ex:
                raise CheckpointError(
                    f"failed to sync ranks after exporting Hugging Face model of step {step_nr}"
                ) from ex
        else:
            if gangs.root.rank == 0:
                done = self._export_op.done()
            else:
                done = True

            try:
                done = broadcast_flag(gangs.root, done)
            except GangError as ex:
                raise CheckpointError(
                    f"failed to broadcast status of Hugging Face model export operation of step {step_nr}"
                ) from ex

            if not done:
                return False

            committer = self._export_op.result()

        self._export_op = None

        committer()

        return True

    @property
    @override
    def step_nr(self) -> int | None:
        return self._step_nr
