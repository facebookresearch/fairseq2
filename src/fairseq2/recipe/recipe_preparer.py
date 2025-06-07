# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import final

from typing_extensions import override

from fairseq2.error import raise_operational_system_error
from fairseq2.file_system import FileSystem
from fairseq2.recipe.logging import DistributedLogConfigurer
from fairseq2.recipe.sweep_tag import SweepTagGenerator
from fairseq2.utils.log import log_config
from fairseq2.utils.structured import ValueConverter
from fairseq2.utils.yaml import YamlDumper
from fairseq2.world_info import WorldInfo


@final
class RecipePreparer:
    def __init__(
        self,
        dir_creator: OutputDirectoryCreator,
        dist_log_configurer: DistributedLogConfigurer,
        config_dumper: RecipeConfigDumper,
    ) -> None:
        self._dir_creator = dir_creator
        self._dist_log_configurer = dist_log_configurer
        self._config_dumper = config_dumper

    def prepare(self, output_dir: Path) -> Path:
        output_dir = self._dir_creator.create(output_dir)

        self._dist_log_configurer.configure(output_dir)

        self._config_dumper.dump(output_dir)

        return output_dir


class OutputDirectoryCreator(ABC):
    @abstractmethod
    def create(self, output_dir: Path) -> Path: ...


@final
class StandardOutputDirectoryCreator(OutputDirectoryCreator):
    def __init__(
        self, sweep_tag_generator: SweepTagGenerator, file_system: FileSystem
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


class RecipeConfigDumper(ABC):
    @abstractmethod
    def dump(self, output_dir: Path) -> None: ...


@final
class StandardRecipeConfigDumper(RecipeConfigDumper):
    def __init__(
        self,
        config: object,
        world_info: WorldInfo,
        value_converter: ValueConverter,
        yaml_dumper: YamlDumper,
    ) -> None:
        self._config = config
        self._world_info = world_info
        self._value_converter = value_converter
        self._yaml_dumper = yaml_dumper

    @override
    def dump(self, output_dir: Path) -> None:
        unstructured_config = self._value_converter.unstructure(self._config)

        log_config("Config", unstructured_config)

        if self._world_info.rank != 0:
            return

        file = output_dir.joinpath("config.yaml")

        try:
            self._yaml_dumper.dump(unstructured_config, file)
        except OSError as ex:
            raise_operational_system_error(ex)
