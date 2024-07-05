# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from argparse import OPTIONAL, ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Generic, Optional, Protocol, TypeVar, final

import yaml
from rich.console import Console
from yaml import YAMLError

from fairseq2.config_registry import ConfigRegistry
from fairseq2.console import get_console, set_console
from fairseq2.logging import get_log_writer
from fairseq2.recipes.logging import setup_basic_logging, setup_logging
from fairseq2.recipes.utils.argparse import BooleanOptionalAction, ConfigAction
from fairseq2.recipes.utils.environment import (
    EnvironmentSetterRegistry,
    default_env_setters,
)
from fairseq2.recipes.utils.log import exception_logger, log_config
from fairseq2.recipes.utils.sweep import SweepTagger, default_sweep_tagger
from fairseq2.typing import DataClass, override
from fairseq2.utils.dataclass import FieldError, dump_dataclass, update_dataclass
from fairseq2.utils.value_converter import ValueConverter, default_value_converter

log = get_log_writer(__name__)


class Cli:
    """Represents the entry point of a command line program."""

    _name: str
    _origin_module: str
    _version: str
    _description: Optional[str]
    _groups: Dict[str, CliGroup]

    def __init__(
        self,
        name: str,
        origin_module: str,
        *,
        version: str,
        description: Optional[str] = None,
    ) -> None:
        """
        :param name:
            The name of the program.
        :param origin_module:
            The name of the origin Python module of the command line program.
        :param version:
            The version of the program.
        :param description:
            The description of the program.
        """
        self._name = name
        self._origin_module = origin_module
        self._version = version
        self._description = description
        self._groups = {}

    def add_group(
        self,
        name: str,
        *,
        origin_module: Optional[str] = None,
        help: Optional[str] = None,
    ) -> CliGroup:
        """Add a command group.

        :param name:
            The name of the command group.
        :param origin_module:
            The name of origin Python module of the command group.
        :param help:
            The help text of the command group.
        """
        if name in self._groups:
            raise ValueError(
                f"`name` must be a unique group name, but '{name}' is already registered."
            )

        group = CliGroup(name, origin_module or self._origin_module, help=help)

        self._groups[group.name] = group

        return group

    def get_group(self, name: str) -> CliGroup:
        """Return the command group of ``name``."""
        try:
            return self._groups[name]
        except KeyError:
            raise ValueError(
                f"`name` must be a registered group name, but is '{name}' instead."
            )

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

    def __call__(self) -> None:
        """Run the program."""
        set_console(Console(highlight=False))

        with exception_logger(log):
            self._run_command()

    def _run_command(self) -> None:
        parser = ArgumentParser(self._name, description=self._description)

        self.init_parser(parser)

        args = parser.parse_args()

        if not hasattr(args, "command"):
            parser.print_usage(sys.stderr)

            sys.exit(2)

        args.command(args)

    @property
    def name(self) -> str:
        """The name of the program."""
        return self._name

    @property
    def origin_module(self) -> str:
        """The name of the origin Python module of the command line program."""
        return self._origin_module

    @property
    def version(self) -> str:
        """The version of the program."""
        return self._version

    @property
    def description(self) -> Optional[str]:
        """The description of the program."""
        return self._description


class CliGroup:
    """Represents a command group of a command line program."""

    _name: str
    _origin_module: str
    _help: Optional[str]
    _groups: Dict[str, CliGroup]
    _commands: Dict[str, CliCommand]

    def __init__(
        self,
        name: str,
        origin_module: str,
        *,
        help: Optional[str] = None,
    ) -> None:
        self._name = name
        self._origin_module = origin_module
        self._help = help
        self._groups = {}
        self._commands = {}

    def add_group(
        self,
        name: str,
        *,
        origin_module: Optional[str] = None,
        help: Optional[str] = None,
    ) -> CliGroup:
        """Add a sub-command group.

        :param name:
            The name of the command group.
        :param origin_module:
            The name of origin Python module of the command group.
        :param help:
            The help text of the command group.
        """
        self._check_name(name)

        group = CliGroup(name, origin_module or self._origin_module, help=help)

        self._groups[group.name] = group

        return group

    def add_command(
        self,
        name: str,
        handler: CliCommandHandler,
        *,
        origin_module: Optional[str] = None,
        help: Optional[str] = None,
    ) -> CliCommand:
        """Add a command.

        :param name:
            The name of the command.
        :param handler:
            The handler of the command.
        :param origin_module:
            The name of origin Python module of the command.
        :param help:
            The help text of the command.
        """
        self._check_name(name)

        command = CliCommand(
            name, handler, origin_module or self._origin_module, help=help
        )

        self._commands[name] = command

        return command

    def _check_name(self, name: str) -> None:
        if name in self._groups:
            raise ValueError(
                f"`name` must be a unique name among groups and commands, but '{name}' is already registered as a group name."
            )

        if name in self._commands:
            raise ValueError(
                f"`name` must be a unique name among groups and commands, but '{name}' is already registered as a command name."
            )

    def get_group(self, name: str) -> CliGroup:
        """Return the sub-command group of ``name``."""
        try:
            return self._groups[name]
        except KeyError:
            raise ValueError(
                f"`name` must be a registered group name, but is '{name}' instead."
            )

    def get_command(self, name: str) -> CliCommand:
        """Return the command of ``name``."""
        try:
            return self._commands[name]
        except KeyError:
            raise ValueError(
                f"`name` must be a registered command name, but is '{name}' instead."
            )

    def init_parser(self, parser: ArgumentParser) -> None:
        """Initialize ``parser`` with command group-specific arguments."""
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

        for command in self._commands.values():
            help = command.help

            if self._origin_module != command.origin_module:
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
    def origin_module(self) -> str:
        """The name of the origin Python module of the command group."""
        return self._origin_module

    @property
    def help(self) -> Optional[str]:
        """The help text of the command group."""
        return self._help


class CliCommand:
    """Represents a command of a command line program."""

    _name: str
    _handler: CliCommandHandler
    _origin_module: str
    _help: Optional[str]

    def __init__(
        self,
        name: str,
        handler: CliCommandHandler,
        origin_module: str,
        *,
        help: Optional[str] = None,
    ) -> None:
        self._name = name
        self._handler = handler
        self._origin_module = origin_module
        self._help = help

    def init_parser(self, parser: ArgumentParser) -> None:
        """Initialize ``parser`` with command group-specific arguments."""
        self._handler.init_parser(parser)

    def __call__(self, args: Namespace) -> None:
        """Run the command."""
        self._handler(args)

    @property
    def name(self) -> str:
        """The name of the command."""
        return self._name

    @property
    def origin_module(self) -> str:
        """The name of the origin Python module of the command."""
        return self._origin_module

    @property
    def help(self) -> Optional[str]:
        """The help text of the command."""
        return self._help


class CliCommandHandler(ABC):
    """Represents the handler of a command of a command line program."""

    @abstractmethod
    def init_parser(self, parser: ArgumentParser) -> None:
        """Initialize ``parser`` with command-specific arguments."""

    @abstractmethod
    def __call__(self, args: Namespace) -> None:
        """Run the command."""


RecipeConfigT = TypeVar("RecipeConfigT", bound=DataClass)

RecipeConfigT_contra = TypeVar(
    "RecipeConfigT_contra", bound=DataClass, contravariant=True
)


class RecipeLoader(Protocol[RecipeConfigT_contra]):
    """Loads a recipe."""

    def __call__(
        self, config: RecipeConfigT_contra, output_dir: Path
    ) -> Callable[[], None]:
        """
        :param name:
            The configuration of the recipe.
        :param output_dir:
            The directory where to store the recipe artifacts.
        """


@final
class RecipeCommandHandler(CliCommandHandler, Generic[RecipeConfigT]):
    """Runs a recipe over command line."""

    _loader: RecipeLoader[RecipeConfigT]
    _preset_configs: ConfigRegistry[RecipeConfigT]
    _default_preset: str
    _env_setters: EnvironmentSetterRegistry
    _value_converter: ValueConverter
    _sweep_tagger: SweepTagger
    _parser: Optional[ArgumentParser]

    def __init__(
        self,
        loader: RecipeLoader[RecipeConfigT],
        preset_configs: ConfigRegistry[RecipeConfigT],
        default_preset: str,
        *,
        env_setters: Optional[EnvironmentSetterRegistry] = None,
        value_converter: Optional[ValueConverter] = None,
        sweep_tagger: Optional[SweepTagger] = None,
    ) -> None:
        """
        :param loader:
            The recipe loader.
        :param preset_configs:
            The registry containing the preset recipe configurations.
        :param default_preset:
            The name of the default preset.
        :param env_setters:
            The registry containing cluster-specific :class:`EnvironmentSetter`
            instances.
        :param value_converter:
            The :class:`ValueConverter` instance to use. If ``None``, the
            default instance will be used.
        :param sweep_tagger:
            The :class:`SweepTagger` instance to use. If ``None``, the default
            instance will be used.
        """
        self._loader = loader
        self._preset_configs = preset_configs
        self._default_preset = default_preset
        self._env_setters = env_setters or default_env_setters
        self._value_converter = value_converter or default_value_converter
        self._sweep_tagger = sweep_tagger or default_sweep_tagger
        self._parser = None

    @override
    def init_parser(self, parser: ArgumentParser) -> None:
        self._parser = parser

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

        clusters = list(self._env_setters.names())

        clusters.sort()

        parser.add_argument(
            "--cluster",
            choices=["auto"] + clusters,
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
    def __call__(self, args: Namespace) -> None:
        console = get_console()

        setup_basic_logging(debug=args.debug)

        assert self._parser is not None

        # If requested, list the preset configurations and exit.
        if args.list_presets:
            if self._preset_configs.names():
                console.print("available presets:")

                for preset in self._preset_configs.names():
                    if preset == self._default_preset:
                        console.print(f"  - {preset} (default)")
                    else:
                        console.print(f"  - {preset}")
            else:
                console.print("no preset configuration found.")

            sys.exit()

        # Load the specified preset configuration.
        try:
            preset_config = self._preset_configs.get(args.preset)
        except ValueError:
            log.error("'{}' is not a valid preset configuration name. Use `--list-presets` to see the available preset configurations.", args.preset)  # fmt: skip

            sys.exit(1)

        config = deepcopy(preset_config)

        # Update the configuration with `--config-file`.
        if args.config_files:
            for config_file in args.config_files:
                try:
                    fp = config_file.open()
                except OSError:
                    log.exception("Configuration file '{}' cannot be read.", config_file)  # fmt: skip

                    sys.exit(1)

                try:
                    config_overrides = yaml.safe_load(fp)
                except (OSError, YAMLError):
                    log.exception("Configuration file '{}' cannot be read.", config_file)  # fmt: skip

                    sys.exit(1)
                finally:
                    fp.close()

                if not isinstance(config_overrides, dict):
                    log.error("Configuration file '{}' must contain a dictionary.", config_file)  # fmt: skip

                    sys.exit(1)

                try:
                    unknown_fields = update_dataclass(
                        config, config_overrides, value_converter=self._value_converter
                    )
                except FieldError as ex:
                    log.exception("Value of the field '{}' in the configuration file '{}' is invalid.", ex.field_name, config_file)  # fmt: skip

                    sys.exit(1)

                if unknown_fields:
                    log.error("Following fields in the configuration file '{}' are unknown: {}", config_file, ", ".join(unknown_fields))  # fmt: skip

                    sys.exit(1)

        # Update the configuration with `--config`.
        if args.config_overrides:
            try:
                unknown_fields = update_dataclass(
                    config, args.config_overrides, value_converter=self._value_converter
                )
            except FieldError as ex:
                log.exception("Value of the field '{}' in `--config` is invalid.", ex.field_name)  # fmt: skip

                sys.exit(1)

            if unknown_fields:
                log.error("Following fields in `--config` are unknown: {}", ", ".join(unknown_fields))  # fmt: skip

                sys.exit(1)

        if args.dump_config:
            dump_dataclass(config, sys.stdout)

            sys.exit()

        # If we are not dumping configuration, `--output-dir` is required.
        if not args.output_dir:
            self._parser.error("the following arguments are required: output_dir")

        self._parser = None

        # Set up cluster-specific environment variables.
        if args.cluster == "auto":
            env_setter = self._env_setters.get_for_inferred_cluster()
        else:
            try:
                env_setter = self._env_setters.get(args.cluster)
            except RuntimeError:
                log.exception("Recipe is not running on a '{}' cluster.", args.cluster)  # fmt: skip

                sys.exit(1)

        try:
            env_setter.set_torch_distributed_env()
        except RuntimeError:
            log.exception("'{}' cluster environment cannot be set.", env_setter.cluster)  # fmt: skip

            sys.exit(1)

        # Determine the output directory.
        if args.no_sweep_dir:
            output_dir = args.output_dir
        else:
            tag = self._sweep_tagger(args.preset, preset_config, config)

            output_dir = args.output_dir.joinpath(tag)

        # Set up distributed logging.
        log_file = output_dir.expanduser().joinpath("logs/rank_{rank}.log").resolve()

        try:
            setup_logging(log_file, debug=args.debug)
        except RuntimeError:
            log.exception("Recipe logging cannot be set up.")

            sys.exit(1)

        log.info("Log files stored under {}.", log_file.parent)

        log_config(config, log, output_dir.joinpath("config.yaml"))

        # Load and run the recipe.
        recipe = self._loader(config, output_dir)

        recipe()
