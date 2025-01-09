# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from argparse import OPTIONAL, ArgumentParser, BooleanOptionalAction, Namespace
from collections.abc import Hashable, Set
from pathlib import Path
from typing import TypeVar, final

from rich.console import Console
from typing_extensions import override

from fairseq2.config_registry import ConfigNotFoundError, ConfigProvider
from fairseq2.context import get_runtime_context
from fairseq2.error import AlreadyExistsError, InvalidOperationError, SetupError
from fairseq2.gang import is_torchrun
from fairseq2.logging import log
from fairseq2.recipes.cluster import (
    ClusterError,
    ClusterHandler,
    ClusterResolver,
    UnknownClusterError,
)
from fairseq2.recipes.logging import DistributedLoggingInitializer, setup_basic_logging
from fairseq2.recipes.runner import (
    ConfigFileNotFoundError,
    ConfigReader,
    EnvironmentBootstrapper,
    Recipe,
    RecipeLoader,
    RecipeRunner,
    StandardConfigReader,
    StandardEnvironmentBootstrapper,
    StandardRecipeRunner,
    SystemSignalHandler,
    get_sweep_keys,
)
from fairseq2.recipes.utils.argparse import ConfigAction
from fairseq2.recipes.utils.rich import get_console, set_console
from fairseq2.recipes.utils.sweep_tagger import (
    NoopSweepTagger,
    StandardSweepTagger,
    SweepFormatError,
    SweepFormatPlaceholderError,
    SweepTagger,
)
from fairseq2.typing import safe_cast
from fairseq2.utils.file import StandardFileSystem
from fairseq2.utils.structured import StructureError, unstructure
from fairseq2.utils.yaml import YamlDumper, YamlError, dump_yaml, load_yaml


class Cli:
    """Represents the entry point of a command line program."""

    _name: str
    _description: str | None
    _origin_module: str
    _version: str
    _groups: dict[str, CliGroup]

    def __init__(
        self,
        name: str,
        origin_module: str,
        *,
        version: str,
        description: str | None = None,
    ) -> None:
        """
        :param name: The name of the program.
        :param origin_module: The name of the origin Python module of the
            command line program.
        :param version: The version of the program.
        :param description: The description of the program.
        """
        self._name = name
        self._description = description
        self._origin_module = origin_module
        self._version = version
        self._groups = {}

    def add_group(
        self,
        name: str,
        *,
        help: str | None = None,
        origin_module: str | None = None,
    ) -> CliGroup:
        """Add a sub-group."""
        group = self._get_or_add_group(name)

        if help is not None:
            group._help = help

        if origin_module is not None:
            group._origin_module = origin_module

        return group

    def get_group(self, name: str) -> CliGroup:
        """Get a sub-group."""
        return self._get_or_add_group(name)

    def _get_or_add_group(self, name: str) -> CliGroup:
        try:
            return self._groups[name]
        except KeyError:
            pass

        group = CliGroup(name, self._origin_module)

        self._groups[name] = group

        return group

    def init_parser(self, parser: ArgumentParser) -> None:
        """Initialize ``parser`` with program-specific arguments."""
        parser.add_argument(
            "--version", action="version", version=f"%(prog)s {self._version}"
        )

        sub_parsers = parser.add_subparsers()

        for group in self._groups.values():
            help = group.help

            if self._origin_module != group.origin_module:
                s = f"origin: {group.origin_module}"

                if help:
                    help = f"{help} ({s})"
                else:
                    help = s

            sub_parser = sub_parsers.add_parser(group.name, help=help)

            group.init_parser(sub_parser)

    def run(self) -> int:
        """Run the program."""
        set_console(Console(highlight=False))

        parser = ArgumentParser(self._name, description=self._description)

        self.init_parser(parser)

        args = parser.parse_args()

        if not hasattr(args, "command"):
            parser.error("no command specified")

        return args.command.run(args)  # type: ignore[no-any-return]

    @property
    def name(self) -> str:
        """The name of the program."""
        return self._name

    @property
    def description(self) -> str | None:
        """The description of the program."""
        return self._description

    @property
    def origin_module(self) -> str:
        """The name of the origin Python module of the command line program."""
        return self._origin_module

    @property
    def version(self) -> str:
        """The version of the program."""
        return self._version


class CliGroup:
    """Represents a command group of a command line program."""

    _name: str
    _groups: dict[str, CliGroup]
    _commands: dict[str, CliCommand]
    _help: str | None
    _origin_module: str

    def __init__(
        self,
        name: str,
        origin_module: str,
        *,
        help: str | None = None,
    ) -> None:
        self._name = name
        self._groups = {}
        self._commands = {}
        self._help = help
        self._origin_module = origin_module

    def add_group(
        self,
        name: str,
        *,
        help: str | None = None,
        origin_module: str | None = None,
    ) -> CliGroup:
        """Add a sub-group."""
        group = self._get_or_add_group(name)
        if group is None:
            raise AlreadyExistsError(
                f"The command group has already a command named '{name}'."
            )

        if help is not None:
            group._help = help

        if origin_module is not None:
            group._origin_module = origin_module

        return group

    def get_group(self, name: str) -> CliGroup:
        """Get a sub-group."""
        group = self._get_or_add_group(name)
        if group is None:
            raise LookupError(
                f"The command group does not have a sub-group named '{name}'."
            ) from None

        return group

    def _get_or_add_group(self, name: str) -> CliGroup | None:
        try:
            return self._groups[name]
        except KeyError:
            pass

        if name in self._commands:
            return None

        group = CliGroup(name, self.origin_module)

        self._groups[name] = group

        return group

    def add_command(
        self,
        name: str,
        handler: CliCommandHandler,
        *,
        help: str | None = None,
        origin_module: str | None = None,
    ) -> CliCommand:
        """Add a command.

        :param name: The name of the command.
        :param handler: The handler of the command.
        :param origin_module: The name of origin Python module of the command.
        :param help: The help text of the command.
        """
        if name in self._groups:
            raise AlreadyExistsError(
                f"The command group has already a sub-group named '{name}'."
            )

        if name in self._commands:
            raise AlreadyExistsError(
                f"The command group has already a command named '{name}'."
            )

        command = CliCommand(
            name, handler, origin_module or self.origin_module, help=help
        )

        self._commands[name] = command

        return command

    def get_command(self, name: str) -> CliCommand:
        """Return the command of ``name``."""
        try:
            return self._commands[name]
        except KeyError:
            raise LookupError(
                f"The command group does not have a command named '{name}'."
            ) from None

    def init_parser(self, parser: ArgumentParser) -> None:
        """Initialize ``parser`` with command group-specific arguments."""
        sub_parsers = parser.add_subparsers()

        for group in self._groups.values():
            help = group.help

            if self.origin_module != group.origin_module:
                s = f"origin: {group.origin_module}"

                if help:
                    help = f"{help} ({s})"
                else:
                    help = s

            sub_parser = sub_parsers.add_parser(group.name, help=help)

            group.init_parser(sub_parser)

        for command in self._commands.values():
            help = command.help

            if self.origin_module != command.origin_module:
                s = f"origin: {command.origin_module}"

                if help:
                    help = f"{help} ({s})"
                else:
                    help = s

            sub_parser = sub_parsers.add_parser(command.name, help=help)

            sub_parser.set_defaults(command=command)

            command.init_parser(sub_parser)

    @property
    def name(self) -> str:
        """The name of the command group."""
        return self._name

    @property
    def help(self) -> str | None:
        """The help text of the command group."""
        return self._help

    @property
    def origin_module(self) -> str:
        """The name of the origin Python module of the command group."""
        return self._origin_module


class CliCommand:
    """Represents a command of a command line program."""

    _name: str
    _handler: CliCommandHandler
    _parser: ArgumentParser | None
    _help: str | None
    _origin_module: str

    def __init__(
        self,
        name: str,
        handler: CliCommandHandler,
        origin_module: str,
        *,
        help: str | None = None,
    ) -> None:
        self._name = name
        self._handler = handler
        self._help = help
        self._origin_module = origin_module

    def init_parser(self, parser: ArgumentParser) -> None:
        """Initialize ``parser`` with command group-specific arguments."""
        self._handler.init_parser(parser)

        self._parser = parser

    def run(self, args: Namespace) -> int:
        """Run the command."""
        if self._parser is None:
            raise InvalidOperationError("`init_parser()` must be called first.")

        try:
            return self._handler.run(self._parser, args)
        finally:
            self._parser = None

    @property
    def name(self) -> str:
        """The name of the command."""
        return self._name

    @property
    def help(self) -> str | None:
        """The help text of the command."""
        return self._help

    @property
    def origin_module(self) -> str:
        """The name of the origin Python module of the command."""
        return self._origin_module


class CliCommandHandler(ABC):
    """Represents the handler of a command of a command line program."""

    @abstractmethod
    def init_parser(self, parser: ArgumentParser) -> None:
        """Initialize ``parser`` with command-specific arguments."""

    @abstractmethod
    def run(self, parser: ArgumentParser, args: Namespace) -> int:
        """Run the command."""


ConfigT = TypeVar("ConfigT")


@final
class RecipeCommandHandler(CliCommandHandler):
    """Runs a recipe over command line."""

    _loader: RecipeLoader[object]
    _preset_configs: ConfigProvider[object]
    _default_preset: str
    _extra_sweep_keys: Set[Hashable] | None

    def __init__(
        self,
        loader: RecipeLoader[ConfigT],
        preset_configs: ConfigProvider[ConfigT],
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

        def untyped_loader(config: object, output_dir: Path) -> Recipe:
            config = safe_cast("config", config, preset_configs.config_kls)

            return loader(config, output_dir)

        self._loader = untyped_loader
        self._preset_configs = preset_configs
        self._default_preset = default_preset
        self._extra_sweep_keys = extra_sweep_keys

    @override
    def init_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--list-presets",
            action="store_true",
            help="list available preset configurations",
        )

        parser.add_argument(
            "--preset",
            default=self._default_preset,
            help="preset configuration (default: %(default)s)",
        )

        parser.add_argument(
            "--config-file",
            dest="config_files",
            metavar="CONFIG_FILE",
            type=Path,
            action="append",
            nargs="*",
            help="yaml configuration file(s)",
        )

        parser.add_argument(
            "--config",
            dest="config_overrides",
            action=ConfigAction,
            help="configuration overrides",
        )

        parser.add_argument(
            "--dump-config",
            action="store_true",
            help="dump the configuration to standard output",
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
    def run(self, parser: ArgumentParser, args: Namespace) -> int:
        if args.list_presets:
            self._print_presets()

            return 0

        setup_basic_logging(debug=args.debug)

        program = self._create_recipe_program(args)

        try:
            program.run(args)

            return 0
        except ConfigNotFoundError as ex:
            parser.error(f"argument --preset: '{ex.name}' is not a known preset configuration. Use `--list-presets` to see the available configurations.")  # fmt: skip
        except ConfigFileNotFoundError as ex:
            parser.error(f"argument --config-file: '{ex.config_file}' does not point to a configuration file")  # fmt: skip
        except MissingOutputDirectoryError:
            parser.error("the following arguments are required: output_dir")
        except UnknownClusterError as ex:
            s = ", ".join(ex.supported_clusters)

            parser.error(f"argument --cluster: '{ex.cluster}' is not a known cluster. Must be one of: auto, none, {s}")  # fmt: skip
        except SweepFormatPlaceholderError as ex:
            s = ", ".join(ex.unknown_keys)

            parser.error(f"argument --sweep-format: must contain only placeholders that correspond to the configuration keys, but contains the following unexpected placeholder(s): {s}")  # fmt: skip
        except SweepFormatError:
            parser.error("argument --sweep-format: must be a non-empty string with brace-enclosed placeholders.")  # fmt: skip
        except ClusterError as ex:
            if ex.cluster == "slurm":
                log.exception("'{}' cluster environment cannot be set. See the logged stack trace for details. If you are within an allocated Slurm job (i.e. `salloc`), make sure to run with `srun`. If you want to run without Slurm, use `--cluster none`.", ex.cluster)  # fmt: skip
            else:
                log.exception("'{}' cluster environment cannot be set. See the logged stack trace for details.", ex.cluster)  # fmt: skip
        except SetupError:
            log.exception("The recipe initialization has failed. See the logged stack trace for details.")  # fmt: skip
        except StructureError:
            log.exception("The recipe configuration cannot be parsed. See the logged stack trace for details.")  # fmt: skip

        return 1

    def _print_presets(self) -> None:
        console = get_console()

        names = self._preset_configs.names()

        if names:
            console.print("available presets:")

            for preset in names:
                if preset == self._default_preset:
                    console.print(f"  - {preset} (default)")
                else:
                    console.print(f"  - {preset}")
        else:
            console.print("no preset configuration found.")

    def _create_recipe_program(self, args: Namespace) -> RecipeProgram:
        file_system = StandardFileSystem()

        config_reader = StandardConfigReader(
            self._preset_configs, file_system, load_yaml
        )

        context = get_runtime_context()

        cluster_handlers = context.get_registry(ClusterHandler)

        cluster_resolver = ClusterResolver(cluster_handlers, is_torchrun=is_torchrun())

        if not args.no_sweep_dir:
            sweep_keys = get_sweep_keys(self._extra_sweep_keys)

            sweep_tagger: SweepTagger = StandardSweepTagger(sweep_keys)
        else:
            sweep_tagger = NoopSweepTagger()

        logging_initializer = DistributedLoggingInitializer()

        env_bootstrapper = StandardEnvironmentBootstrapper(
            cluster_resolver, sweep_tagger, file_system, logging_initializer, dump_yaml
        )

        signal_handler = SystemSignalHandler()

        runner = StandardRecipeRunner(self._loader, signal_handler)

        return RecipeProgram(config_reader, env_bootstrapper, runner, dump_yaml)


@final
class RecipeProgram:
    _config_reader: ConfigReader
    _env_bootstrapper: EnvironmentBootstrapper
    _runner: RecipeRunner
    _yaml_dumper: YamlDumper

    def __init__(
        self,
        config_reader: ConfigReader,
        env_bootstrapper: EnvironmentBootstrapper,
        runner: RecipeRunner,
        yaml_dumper: YamlDumper,
    ) -> None:
        self._config_reader = config_reader
        self._env_bootstrapper = env_bootstrapper
        self._runner = runner
        self._yaml_dumper = yaml_dumper

    def run(self, args: Namespace) -> None:
        config = self._config_reader.read(
            args.preset, args.config_files, args.config_overrides
        )

        if args.dump_config:
            unstructured_config = unstructure(config)

            try:
                self._yaml_dumper(unstructured_config, sys.stdout)
            except YamlError as ex:
                raise SetupError(
                    "The recipe configuration cannot be dumped to stdout. See the nested exception for details."
                ) from ex

            return

        if not args.output_dir:
            raise MissingOutputDirectoryError("`args.output_dir` must be specified.")

        output_dir = self._env_bootstrapper.run(
            args.preset,
            config,
            args.output_dir,
            cluster=args.cluster,
            sweep_format=args.sweep_format,
            debug=args.debug,
        )

        self._runner.run(config, output_dir)


class MissingOutputDirectoryError(ValueError):
    pass
