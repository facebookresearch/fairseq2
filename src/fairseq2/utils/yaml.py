# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, TypeAlias, final

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError
from typing_extensions import override

from fairseq2.file_system import FileMode, FileSystem


class YamlLoader(ABC):
    @abstractmethod
    def load(self, input_: Path | IO[str]) -> list[object]:
        """
        :raises YamlError:
        :raises OSError:
        """


class YamlDumper(ABC):
    @abstractmethod
    def dump(self, obj: object, output: Path | IO[str]) -> None:
        """
        :raises YamlError:
        :raises OSError:
        """


YamlError: TypeAlias = YAMLError


@final
class RuamelYamlLoader(YamlLoader):
    def __init__(self, file_system: FileSystem) -> None:
        yaml = YAML(typ="safe", pure=True)

        self._yaml = yaml
        self._file_system = file_system

    @override
    def load(self, input_: Path | IO[str]) -> list[object]:
        if isinstance(input_, Path):
            fp = self._file_system.open_text(input_)
            with fp:
                return self.load(fp)

        it = self._yaml.load_all(input_)

        return list(it)


@final
class RuamelYamlDumper(YamlDumper):
    def __init__(self, file_system: FileSystem) -> None:
        yaml = YAML(typ="safe", pure=True)

        yaml.default_flow_style = False
        yaml.sort_base_mapping_type_on_output = False  # type: ignore[assignment]

        self._yaml = yaml
        self._file_system = file_system

    @override
    def dump(self, obj: object, output: Path | IO[str]) -> None:
        if isinstance(output, Path):
            fp = self._file_system.open_text(output, mode=FileMode.WRITE)
            with fp:
                self.dump(obj, fp)
        else:
            self._yaml.dump(obj, output)
