# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import shlex
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import ExitStack
from pathlib import Path
from subprocess import CompletedProcess
from typing import Final, Protocol, final

from torch.utils.hooks import RemovableHandle
from typing_extensions import override

from fairseq2.error import OperationalError
from fairseq2.file_system import FileMode, FileSystem
from fairseq2.gang import GangError, Gangs, broadcast_flag, raise_operational_gang_error
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
    def maybe_complete_operation(self, *, blocking: bool = False) -> bool | None: ...

    @property
    @abstractmethod
    def is_exporting(self) -> bool: ...

    @abstractmethod
    def register_export_hook(self, hook: CheckpointHGExportHook) -> RemovableHandle: ...


class CheckpointHGExportHook(Protocol):
    def __call__(self, step_nr: int, export_dir: Path) -> None: ...


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
        self._export_hooks: dict[int, CheckpointHGExportHook] = OrderedDict()

    @override
    def maybe_complete_operation(self, *, blocking: bool = False) -> bool | None:
        return None

    @property
    @override
    def is_exporting(self) -> bool:
        return False

    @override
    def register_export_hook(self, hook: CheckpointHGExportHook) -> RemovableHandle:
        handle = RemovableHandle(self._export_hooks)

        self._export_hooks[handle.id] = hook

        return handle


NOOP_CHECKPOINT_HG_EXPORTER: Final = _NoopCheckpointHGExporter()


@final
class OutOfProcCheckpointHGExporter(CheckpointHGExporter):
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
        self._export_hooks: dict[int, CheckpointHGExportHook] = OrderedDict()

    @override
    def export(
        self,
        step_nr: int,
        *,
        exported_callback: Callable[[int], None] | None = None,
        blocking: bool = False,
    ) -> None:
        self.maybe_complete_operation(blocking=True)

        def do_export() -> Callable[[], None]:
            export_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}/hg")

            args = [sys.executable, "-m", "fairseq2.models.utils.hg_export", "--no-rich", "--checkpoint-dir", str(self._checkpoint_dir), f"checkpoint_step_{step_nr}", str(export_dir)]  # fmt: skip

            if self._gangs.root.rank == 0:
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
            else:
                result = CompletedProcess(args, returncode=0)

            def commit() -> None:
                if result.returncode != 0:
                    log.warning("Hugging Face export operation of step {} failed. See operation output at {}.", step_nr, err_file)  # fmt: skip

                    return

                if exported_callback is not None:
                    exported_callback(step_nr)

                for hook in self._export_hooks.values():
                    hook(step_nr, export_dir)

            return commit

        if blocking:
            committer = do_export()

            committer()
        else:
            try:
                self._export_op = self._thread_pool.queue(do_export)
            except RuntimeError as ex:
                raise OperationalError("A thread pool queue operation failed.") from ex

    @override
    def maybe_complete_operation(self, *, blocking: bool = False) -> bool | None:
        if self._export_op is None:
            return None

        gangs = self._gangs

        if blocking:
            committer = self._export_op.result()

            try:
                gangs.root.barrier()
            except GangError as ex:
                raise_operational_gang_error(ex)
        else:
            if gangs.root.rank == 0:
                done = self._export_op.done()
            else:
                done = True

            try:
                done = broadcast_flag(gangs.root, done)
            except GangError as ex:
                raise_operational_gang_error(ex)

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

    @override
    def register_export_hook(self, hook: CheckpointHGExportHook) -> RemovableHandle:
        handle = RemovableHandle(self._export_hooks)

        self._export_hooks[handle.id] = hook

        return handle
