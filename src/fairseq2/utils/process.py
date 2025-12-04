# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from subprocess import CompletedProcess
from typing import BinaryIO, TextIO, final

from typing_extensions import override


class ProcessRunner(ABC):
    @abstractmethod
    def run(
        self,
        args: Sequence[str],
        stdout: BinaryIO | None = None,
        stderr: BinaryIO | None = None,
        capture_output: bool = False,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CompletedProcess[bytes]: ...

    @abstractmethod
    def run_text(
        self,
        args: Sequence[str],
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        capture_output: bool = False,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CompletedProcess[str]: ...


@final
class StandardProcessRunner(ProcessRunner):
    @override
    def run(
        self,
        args: Sequence[str],
        stdout: BinaryIO | None = None,
        stderr: BinaryIO | None = None,
        capture_output: bool = False,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CompletedProcess[bytes]:
        return subprocess.run(
            args,
            stdout=stdout,
            stderr=stderr,
            capture_output=capture_output,
            cwd=cwd,
            env=env,
        )

    @override
    def run_text(
        self,
        args: Sequence[str],
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        capture_output: bool = False,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CompletedProcess[str]:
        return subprocess.run(
            args,
            stdout=stdout,
            stderr=stderr,
            capture_output=capture_output,
            text=True,
            cwd=cwd,
            env=env,
        )
