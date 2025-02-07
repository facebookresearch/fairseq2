# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import sys
from argparse import OPTIONAL, ArgumentParser, BooleanOptionalAction, Namespace
from collections.abc import Callable, Hashable, Iterable, Mapping, Set
from itertools import chain
from logging import getLogger
from pathlib import Path
from signal import SIGUSR1, signal
from types import FrameType
from typing import Protocol, TypeAlias, final, runtime_checkable

from typing_extensions import override

from fairseq2.cli import CliArgumentError, CliCommandHandler, setup_logging
from fairseq2.cli.utils.argparse import ConfigAction
from fairseq2.cli.utils.cluster import set_torch_distributed_variables
from fairseq2.cli.utils.rich import create_rich_progress_reporter, get_console
from fairseq2.config_registry import ConfigNotFoundError, ConfigProvider
from fairseq2.context import RuntimeContext
from fairseq2.error import ContractError, ProgramError
from fairseq2.logging import LoggingSetupError, log
from fairseq2.recipes.cluster import ClusterError, UnknownClusterError
from fairseq2.recipes.logging import DistributedLoggingInitializer
from fairseq2.recipes.utils.log import log_config
from fairseq2.recipes.utils.progress import ProgressReporter
from fairseq2.recipes.utils.sweep_tag import (
    SweepFormatError,
    SweepFormatPlaceholderError,
    SweepTagGenerator,
    get_sweep_keys,
)
from fairseq2.utils.env import InvalidEnvironmentVariableError, get_rank, get_world_size
from fairseq2.utils.file import FileSystem
from fairseq2.utils.merge import MergeError, merge_object, to_mergeable
from fairseq2.utils.structured import StructureError, unstructure
from fairseq2.utils.yaml import (
    StandardYamlDumper,
    StandardYamlLoader,
    YamlDumper,
    YamlError,
    YamlLoader,
)


@final
class RecipeCommandHandler(CliCommandHandler):
    """Runs a recipe over command line."""

    _loader: RecipeLoader
    _config_kls: type[object]
    _default_preset: str
    _extra_sweep_keys: Set[Hashable] | None

    def __init__(
        self,
        loader: RecipeLoader,
        config_kls: type[object],
        default_preset: str,
        *,
        extra_sweep_keys: Set[Hashable] | None = None,
    ) -> None:
        """
        :param loader: The recipe loader.
        :param preset_configs: The registry containing the preset recipe
            configurations.
        :param default_preset: The name of the default preset.
        :param extra_sweep_keys: The recipe specific configuration keys to
            include in the sweep directory name.
        """
        self._loader = loader
        self._config_kls = config_kls
        self._default_preset = default_preset
        self._extra_sweep_keys = extra_sweep_keys

    @override
    def init_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--list-preset-configs",
            action="store_true",
            help="list available preset configurations",
        )

        parser.add_argument(
            "--preset",
            default=self._default_preset,
            help="preset configuration name (default: %(default)s)",
        )

        parser.add_argument(
            "--config-file",
            dest="config_override_files",
            metavar="CONFIG_FILE",
            type=Path,
            action="append",
            nargs="*",
            help="configuration file(s)",
        )

        parser.add_argument(
            "--config",
            dest="config_overrides",
            action=ConfigAction,
            help="command line configuration overrides",
        )

        parser.add_argument(
            "--dump-config",
            action="store_true",
            help="dump the configuration in mergeable format to standard output",
        )

        parser.add_argument(
            "--no-sweep-dir",
            action="store_true",
            help="do not create sweep directory",
        )

        parser.add_argument(
            "--sweep-format",
            default="ps_{preset}.ws_{world_size}.{hash}",
            help="format of the sweep directory name (default: %(default)s)",
        )

        parser.add_argument(
            "--cluster",
            default="auto",
            help="cluster on which the recipe runs (default: %(default)s)",
        )

        parser.add_argument(
            "--debug",
            action=BooleanOptionalAction,
            help="log at debug level",
        )

        parser.add_argument(
            "output_dir",
            type=Path,
            nargs=OPTIONAL,
            help="directory to store recipe artifacts",
        )

    @override
    def run(
        self, context: RuntimeContext, parser: ArgumentParser, args: Namespace
    ) -> int:
        if args.list_preset_configs:
            self._print_preset_configs(context)

            return 0

        setup_logging(debug=args.debug)

        try:
            config = self._read_config(context, args)
        except ConfigNotFoundError as ex:
            raise CliArgumentError(
                "preset", f"'{ex.name}' is not a known preset name. Use `--list-preset-configs` to see the available configurations."  # fmt: skip
            ) from None
        except ConfigFileNotFoundError as ex:
            raise CliArgumentError(
                "--config-file", f"{ex.config_file} does not point to a configuration file."  # fmt: skip
            ) from None
        except InvalidConfigFileError as ex:
            raise CliArgumentError(
                "--config-file", f"{ex.config_file} does not contain a valid configuration override. See logged stack trace for details."  # fmt: skip
            ) from ex
        except InvalidConfigOverrideError as ex:
            raise CliArgumentError(
                "--config-file", "key-value pair(s) cannot be applied over the preset configuration. See logged stack trace for details."  # fmt: skip
            ) from ex
        except ConfigReadError as ex:
            raise ProgramError(
                "The recipe configuration cannot be read. See the nested exception for details."
            ) from ex

        if args.dump_config:
            if isinstance(config, Mapping):
                config = to_mergeable(config)

            yaml_dumper = StandardYamlDumper(context.file_system)

            try:
                yaml_dumper.dump(config, sys.stdout)
            except YamlError as ex:
                raise ProgramError(
                    "The recipe configuration cannot be dumped to stdout. See the nested exception for details."
                ) from ex

            return 0

        if not args.output_dir:
            raise CliArgumentError("output_dir", "required")

        try:
            set_torch_distributed_variables(context, args.cluster)
        except UnknownClusterError as ex:
            s = ", ".join(ex.supported_clusters)

            raise CliArgumentError(
                "cluster", f"'{ex.cluster}' is not a known cluster. Must be one of: auto, none, {s}"  # fmt: skip
            ) from None
        except ClusterError as ex:
            if ex.cluster == "slurm":
                message = f"'{ex.cluster}' cluster environment cannot be set. See logged stack trace for details. If you are within an allocated Slurm job (i.e. `salloc`), make sure to run with `srun`. If you want to run without Slurm, use `--cluster none`."
            else:
                message = f"'{ex.cluster}' cluster environment cannot be set. See logged stack trace for details."

            raise ProgramError(message) from ex

        output_dir: Path = args.output_dir

        try:
            sweep_tag = self._create_sweep_tag(context, args, config)
        except SweepFormatPlaceholderError as ex:
            s = ", ".join(ex.unknown_keys)

            raise CliArgumentError(
                "--sweep-format", f"must contain only placeholders that correspond to the configuration keys, but contains the following unexpected placeholder(s): {s}"  # fmt: skip
            ) from None
        except SweepFormatError:
            raise CliArgumentError(
                "--sweep-format", "must be a non-empty string with brace-enclosed placeholders."  # fmt: skip
            ) from None
        except SweepTagError as ex:
            raise ProgramError(
                "The sweep tag cannot be generated. See the nested exception for details."
            ) from ex

        output_dir = self._create_output_directory(context, output_dir, sweep_tag)

        self._setup_distributed_logging(context, output_dir)

        self._dump_config(context, config, output_dir)

        try:
            recipe = self._loader(context, config, output_dir)
        except StructureError as ex:
            raise CliArgumentError(
                None, "The recipe configuration cannot be parsed. See logged stack trace for details."  # fmt: skip
            ) from ex

        # If the recipe is stoppable, use SIGUSR1 as the stop signal.
        if isinstance(recipe, Stoppable):

            def request_stop(signum: int, frame: FrameType | None) -> None:
                log.info("SIGUSR1 received. Requesting recipe to stop.")

                recipe.request_stop()

            signal(SIGUSR1, request_stop)

        progress_reporter = create_rich_progress_reporter()

        recipe(progress_reporter)

        return 0

    def _print_preset_configs(self, context: RuntimeContext) -> None:
        console = get_console()

        configs = context.get_config_registry(self._config_kls)

        preset_names = configs.names()

        if preset_names:
            console.print("available presets:")

            for preset in preset_names:
                if preset == self._default_preset:
                    console.print(f"  - {preset} (default)")
                else:
                    console.print(f"  - {preset}")
        else:
            console.print("no preset configuration found.")

    def _read_config(self, context: RuntimeContext, args: Namespace) -> object:
        configs = context.get_config_registry(self._config_kls)

        file_system = context.file_system

        yaml_loader = StandardYamlLoader(file_system)

        config_reader = ConfigReader(configs, file_system, yaml_loader)

        if args.config_override_files:
            config_override_files = chain.from_iterable(args.config_override_files)
        else:
            config_override_files = None

        return config_reader.read(
            args.preset, config_override_files, args.config_overrides
        )

    def _create_sweep_tag(
        self, context: RuntimeContext, args: Namespace, config: object
    ) -> str | None:
        if args.no_sweep_dir:
            return None

        try:
            world_size = get_world_size(os.environ)
        except InvalidEnvironmentVariableError as ex:
            raise SweepTagError(
                "The world size cannot be determined. See the nested exception for details."
            ) from ex

        keys = get_sweep_keys(self._extra_sweep_keys)

        tag_generator = SweepTagGenerator(world_size, keys, args.sweep_format)

        return tag_generator.generate(args.preset, config)

    @staticmethod
    def _create_output_directory(
        context: RuntimeContext, output_dir: Path, sweep_tag: str | None
    ) -> Path:
        if sweep_tag is not None:
            output_dir = output_dir.joinpath(sweep_tag)

        try:
            context.file_system.make_directory(output_dir)
        except OSError as ex:
            raise ProgramError(
                f"The '{output_dir}' recipe directory cannot be created. See the nested exception for details."
            ) from ex

        return output_dir

    @staticmethod
    def _setup_distributed_logging(context: RuntimeContext, output_dir: Path) -> None:
        logger = getLogger()

        initializer = DistributedLoggingInitializer(
            logger, os.environ, context.file_system
        )

        try:
            initializer.initialize(output_dir)
        except LoggingSetupError as ex:
            raise ProgramError(
                "The distributed logging setup has failed. See the nested exception for details."
            ) from ex

        log.info("Log files are stored under {}.", output_dir)

    @staticmethod
    def _dump_config(context: RuntimeContext, config: object, output_dir: Path) -> None:
        yaml_dumper = StandardYamlDumper(context.file_system)

        dumper = ConfigDumper(os.environ, yaml_dumper)

        try:
            dumper.dump(config, output_dir)
        except ConfigDumpError as ex:
            raise ProgramError(
                "The recipe configuration cannot be saved. See the nested exception for details."
            ) from ex


class SweepTagError(Exception):
    pass


Recipe: TypeAlias = Callable[[ProgressReporter], None]


class RecipeLoader(Protocol):
    def __call__(
        self, context: RuntimeContext, config: object, output_dir: Path
    ) -> Recipe: ...


@runtime_checkable
class Stoppable(Protocol):
    """Represents a recipe that supports graceful stopping."""

    def request_stop(self) -> None: ...


@final
class ConfigReader:
    _configs: ConfigProvider[object]
    _file_system: FileSystem
    _yaml_loader: YamlLoader

    def __init__(
        self,
        configs: ConfigProvider[object],
        file_system: FileSystem,
        yaml_loader: YamlLoader,
    ) -> None:
        self._configs = configs
        self._file_system = file_system
        self._yaml_loader = yaml_loader

    def read(
        self,
        preset: str,
        config_override_files: Iterable[Path] | None,
        config_overrides: Iterable[Mapping[str, object]] | None,
    ) -> object:
        # Load the preset configuration.
        preset_config = self._configs.get(preset)

        try:
            unstructured_config = unstructure(preset_config)
        except StructureError as ex:
            raise ContractError(
                f"The '{preset}' preset configuration cannot be unstructured. See the nested exception for details."
            ) from ex

        # Update the configuration with `--config-override-file`.
        if config_override_files:
            for config_override_file in config_override_files:
                try:
                    is_file = self._file_system.is_file(config_override_file)
                except OSError as ex:
                    raise ConfigReadError(
                        f"The '{config_override_file}' configuration file cannot be read. See the nested exception for details."
                    ) from ex

                if not is_file:
                    raise ConfigFileNotFoundError(config_override_file)

                try:
                    unstructured_config_overrides = self._yaml_loader.load(
                        config_override_file
                    )
                except YamlError as ex:
                    raise InvalidConfigFileError(
                        config_override_file, f"The '{config_override_file}' configuration file cannot be loaded. See the nested exception for details."  # fmt: skip
                    ) from ex
                except OSError as ex:
                    raise ConfigReadError(
                        f"The '{config_override_file}' configuration file cannot be read. See the nested exception for details."
                    ) from ex

                if len(unstructured_config_overrides) == 0:
                    raise InvalidConfigFileError(
                        config_override_file, f"The '{config_override_file}' does not contain any YAML document."  # fmt: skip
                    )

                try:
                    unstructured_config = merge_object(
                        unstructured_config, unstructured_config_overrides[0]
                    )
                except MergeError as ex:
                    raise InvalidConfigFileError(
                        config_override_file, f"The '{config_override_file}' configuration file cannot be merged with the preset configuration. See the nested exception for details."  # fmt: skip
                    ) from ex

        # Update the configuration with `--config`.
        if config_overrides:
            for overrides in config_overrides:
                try:
                    unstructured_config = merge_object(unstructured_config, overrides)
                except MergeError as ex:
                    raise InvalidConfigOverrideError(
                        "The configuration overrides cannot be merged with the preset recipe configuration. See the nested exception for details."
                    ) from ex

        return unstructured_config


class ConfigReadError(Exception):
    pass


class ConfigFileNotFoundError(Exception):
    config_file: Path

    def __init__(self, config_file: Path) -> None:
        super().__init__(
            f"The '{config_file}' path does not point to a configuration file."
        )

        self.config_file = config_file


class InvalidConfigFileError(Exception):
    config_file: Path

    def __init__(self, config_file: Path, message: str) -> None:
        super().__init__(message)

        self.config_file = config_file


class InvalidConfigOverrideError(Exception):
    pass


@final
class ConfigDumper:
    _env: Mapping[str, str]
    _yaml_dumper: YamlDumper

    def __init__(self, env: Mapping[str, str], yaml_dumper: YamlDumper) -> None:
        self._env = env
        self._yaml_dumper = yaml_dumper

    def dump(self, config: object, output_dir: Path) -> None:
        log_config(log, "Config", config)

        try:
            rank = get_rank(self._env)
        except InvalidEnvironmentVariableError as ex:
            raise ConfigDumpError(
                "The rank of the process cannot be determined. See the nested exception for details."
            ) from ex

        if rank != 0:
            return

        file = output_dir.joinpath("config.yaml")

        if isinstance(config, Mapping):
            config = to_mergeable(config)

        try:
            self._yaml_dumper.dump(config, file)
        except OSError as ex:
            raise ConfigDumpError(
                f"The configuration cannot be saved to the '{file}' file. See the nested exception for details."
            ) from ex


class ConfigDumpError(Exception):
    pass
