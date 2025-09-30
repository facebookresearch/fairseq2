# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from subprocess import DEVNULL, CalledProcessError
from typing import Any, Final, final

from typing_extensions import override

from fairseq2.error import InvalidOperationError, OperationalError
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
    def __init__(self, output_dir: Path, thread_pool: ThreadPool) -> None:
        self._checkpoint_dir = output_dir.joinpath("checkpoints")
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

            args: Any = ["python", "-m", "fairseq2.models.utils.hg_export", "--checkpoint-dir", self._checkpoint_dir, f"checkpoint_step_{step_nr}", export_dir]  # fmt: skip

            try:
                subprocess.run(args, stdout=DEVNULL, stderr=DEVNULL, check=True)
            except CalledProcessError as ex:
                raise OperationalError(
                    f"Background process failed while exporting the Hugging Face model of step {step_nr}."
                ) from ex

            if exported_callback is not None:
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
