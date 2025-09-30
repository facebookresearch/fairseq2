# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import final

from fairseq2.error import raise_operational_system_error
from fairseq2.file_system import FileSystem
from fairseq2.recipe.internal.sweep_tag import _SweepTagGenerator


@final
class _OutputDirectoryCreator:
    def __init__(
        self, sweep_tag_generator: _SweepTagGenerator, file_system: FileSystem
    ) -> None:
        self._sweep_tag_generator = sweep_tag_generator
        self._file_system = file_system

    def create(self, output_dir: Path) -> Path:
        tag = self._sweep_tag_generator.maybe_generate()
        if tag is not None:
            output_dir = output_dir.joinpath(tag)

        try:
            output_dir = self._file_system.resolve(output_dir)

            self._file_system.make_directory(output_dir)
        except OSError as ex:
            raise_operational_system_error(ex)

        return output_dir
