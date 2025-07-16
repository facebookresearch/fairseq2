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
from typing import Any, final

from typing_extensions import override

from fairseq2.error import InvalidOperationError
from fairseq2.utils.threading import ThreadPool


class HuggingFaceSaver(ABC):
    @abstractmethod
    def save(
        self,
        step_nr: int,
        *,
        callback: Callable[[int], None] | None = None,
        blocking: bool = False,
    ) -> None: ...

    @abstractmethod
    def complete_pending(self) -> None: ...

    @property
    @abstractmethod
    def is_saving(self) -> bool: ...


@final
class OutOfProcHuggingFaceSaver(HuggingFaceSaver):
    _checkpoint_dir: Path
    _thread_pool: ThreadPool
    _save_op: Future[None] | None

    def __init__(self, checkpoint_dir: Path, thread_pool: ThreadPool) -> None:
        self._checkpoint_dir = checkpoint_dir

        self._thread_pool = thread_pool

        self._save_op = None

    @override
    def save(
        self,
        step_nr: int,
        *,
        callback: Callable[[int], None] | None = None,
        blocking: bool = False,
    ) -> None:
        if self._save_op is not None:
            if not self._save_op.done():
                raise InvalidOperationError(
                    "A Hugging Face save operation is already in progress."
                )

            self._save_op = None

        def do_save() -> None:
            save_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}/hg")

            args: Any = ["fairseq2", "convert", "fs2_to_hg", "--checkpoint-dir", self._checkpoint_dir, f"checkpoint_step_{step_nr}", save_dir]  # fmt: skip

            try:
                subprocess.run(args, stdout=DEVNULL, stderr=DEVNULL, check=True)
            except CalledProcessError as ex:
                raise HuggingFaceSaveError(
                    step_nr, f"The background process has failed while saving the Hugging Face model of step {step_nr}. See the nested exception for details."  # fmt: skip
                ) from ex

            if callback is not None:
                callback(step_nr)

        if blocking:
            do_save()
        else:
            self._save_op = self._thread_pool.queue(do_save)

    @override
    def complete_pending(self) -> None:
        if self._save_op is None:
            return

        self._save_op.result()

        self._save_op = None

    @property
    @override
    def is_saving(self) -> bool:
        if self._save_op is not None:
            return not self._save_op.done()

        return False


class HuggingFaceSaveError(Exception):
    step_nr: int

    def __init__(self, step_nr: int, message: str) -> None:
        super().__init__(message)

        self.step_nr = step_nr
