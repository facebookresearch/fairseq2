# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, TypeAlias, final

import yaml
from typing_extensions import override
from yaml import YAMLError

from fairseq2.utils.file import FileMode, FileSystem


class YamlLoader(ABC):
    @abstractmethod
    def load(self, input_: Path | IO[str]) -> list[object]: ...


class YamlDumper(ABC):
    @abstractmethod
    def dump(self, obj: object, output: Path | IO[str]) -> None: ...


YamlError: TypeAlias = YAMLError


@final
class StandardYamlLoader(YamlLoader):
    _file_system: FileSystem

    def __init__(self, file_system: FileSystem) -> None:
        self._file_system = file_system

    @override
    def load(self, input_: Path | IO[str]) -> list[object]:
        if isinstance(input_, Path):
            fp = self._file_system.open_text(input_)

            try:
                return self.load(fp)
            finally:
                fp.close()

        itr = yaml.safe_load_all(input_)

        return list(itr)


@final
class StandardYamlDumper(YamlDumper):
    _file_system: FileSystem

    def __init__(self, file_system: FileSystem) -> None:
        self._file_system = file_system

    @override
    def dump(self, obj: object, output: Path | IO[str]) -> None:
        if isinstance(output, Path):
            fp = self._file_system.open_text(output, mode=FileMode.WRITE)

            try:
                self.dump(obj, fp)
            finally:
                fp.close()
        else:
            yaml.safe_dump(obj, output, sort_keys=False)


def read_yaml(s: str) -> object:
    return yaml.safe_load(s)
