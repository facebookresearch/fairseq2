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
from itertools import chain
from pathlib import Path
from typing import Generic, Mapping, Sequence, TypeVar, final

from rich.console import Console
from typing_extensions import override

from fairseq2.config_registry import ConfigNotFoundError, ConfigProvider
from fairseq2.error import (
    AlreadyExistsError,
    ContractError,
    InvalidOperationError,
    SetupError,
)
from fairseq2.logging import log
from fairseq2.recipes.cluster import ClusterError, UnknownClusterError
from fairseq2.recipes.console import get_console, set_console
from fairseq2.recipes.logging import setup_basic_logging
from fairseq2.recipes.runner import RecipeLoader, run_recipe
from fairseq2.recipes.utils.argparse import ConfigAction
from fairseq2.recipes.utils.sweep_tagger import (
    SweepFormatError,
    SweepFormatPlaceholderError,
)
from fairseq2.typing import DataClass
from fairseq2.utils.structured import (
    StructureError,
    merge_unstructured,
    structure,
    unstructure,
)
from fairseq2.utils.yaml import YamlError, dump_yaml, load_yaml


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
            return self._handler.run(args)
        except CliArgumentError as ex:
            self._parser.error(str(ex))
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
    def run(self, args: Namespace) -> int:
        """Run the command."""


class CliArgumentError(Exception):
    def __init__(self, argument_name: str | None, message: str) -> None:
        if argument_name is not None:
            message = f"argument {argument_name}: {message}"

        super().__init__(message)


RecipeConfigT = TypeVar("RecipeConfigT", bound=DataClass)


@final
class RecipeCommandHandler(CliCommandHandler, Generic[RecipeConfigT]):
    """Runs a recipe over command line."""

    _loader: RecipeLoader[RecipeConfigT]
    _preset_configs: ConfigProvider[RecipeConfigT]
    _default_preset: str
    _extra_sweep_keys: Set[Hashable] | None

    def __init__(
        self,
        loader: RecipeLoader[RecipeConfigT],
        preset_configs: ConfigProvider[RecipeConfigT],
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
    def run(self, args: Namespace) -> int:
        try:
            self._do_run(args)

            return 0
        except ClusterError as ex:
            log.exception("'{}' cluster environment cannot be set. See the logged stack trace for details.", ex.cluster)  # fmt: skip
        except SetupError:
            log.exception("The recipe initialization has failed. See the logged stack trace for details.")  # fmt: skip
        except StructureError:
            log.exception("The recipe configuration cannot be parsed. See the logged stack trace for details.")  # fmt: skip

        return 1

    def _do_run(self, args: Namespace) -> None:
        setup_basic_logging(debug=args.debug)

        if args.list_presets:
            self._list_presets()

            return

        preset_configs = self._preset_configs

        try:
            unstructured_config = read_unstructured_config(
                preset_configs, args.preset, args.config_files, args.config_overrides
            )
        except ConfigNotFoundError:
            raise CliArgumentError(
                "--preset", f"'{args.preset}' is not a known preset configuration. Use `--list-presets` to see the available configurations."  # fmt: skip
            ) from None
        except ConfigFileNotFoundError as ex:
            raise CliArgumentError(
                "--config-file", f"'{ex.config_file}' does not point to a configuration file"  # fmt: skip
            ) from None

        if args.dump_config:
            try:
                dump_yaml(unstructured_config, sys.stdout)
            except OSError as ex:
                raise SetupError(
                    "The recipe configuration cannot be dumped to stdout. See the nested exception for details."
                ) from ex

            return

        if not args.output_dir:
            raise CliArgumentError(
                None, "the following arguments are required: output_dir"
            )

        config = structure(unstructured_config, preset_configs.config_kls)

        try:
            run_recipe(
                self._loader,
                args.preset,
                config,
                args.output_dir,
                cluster=args.cluster,
                no_sweep_dir=args.no_sweep_dir,
                sweep_format=args.sweep_format,
                extra_sweep_keys=self._extra_sweep_keys,
                debug=args.debug,
            )
        except UnknownClusterError:
            raise CliArgumentError(
                "--cluster", f"'{args.cluster}' is not a known cluster."
            ) from None
        except SweepFormatPlaceholderError as ex:
            s = ", ".join(ex.unknown_keys)

            raise CliArgumentError(
                "--sweep-format", f"must contain only placeholders that correspond to the configuration keys, but contains the following unexpected placeholder(s): {s}"  # fmt: skip
            ) from None
        except SweepFormatError:
            raise CliArgumentError(
                "--sweep-format", "must be a non-empty string with brace-enclosed placeholders."  # fmt: skip
            ) from None

    def _list_presets(self) -> None:
        console = get_console()

        presets = self._preset_configs.names()

        if presets:
            console.print("available presets:")

            for preset in presets:
                if preset == self._default_preset:
                    console.print(f"  - {preset} (default)")
                else:
                    console.print(f"  - {preset}")
        else:
            console.print("no preset configuration found.")


def read_unstructured_config(
    preset_configs: ConfigProvider[object],
    preset: str,
    config_files: Sequence[Sequence[Path]] | None,
    config_overrides: Sequence[Mapping[str, object]] | None,
) -> object:
    # Load the preset configuration.
    preset_config = preset_configs.get(preset)

    try:
        unstructured_config = unstructure(preset_config)
    except StructureError as ex:
        raise ContractError(
            f"The '{preset}' preset configuration cannot be unstructured. See the nested exception for details."
        ) from ex

    # Update the configuration with `--config-file`.
    if config_files:
        for config_file in chain.from_iterable(config_files):
            if not config_file.exists() or not config_file.is_file():
                raise ConfigFileNotFoundError(config_file)

            try:
                unstructured_config_overrides = load_yaml(config_file)
            except YamlError as ex:
                raise StructureError(
                    f"The '{config_file}' configuration file cannot be merged with the preset configuration. See the nested exception for details."
                ) from ex
            except OSError as ex:
                raise SetupError(
                    f"The '{config_file}' configuration file cannot be read. See the nested exception for details."
                ) from ex

            try:
                unstructured_config = merge_unstructured(
                    unstructured_config, unstructured_config_overrides[0]
                )
            except StructureError as ex:
                raise StructureError(
                    f"The '{config_file}' configuration file cannot be merged with the preset configuration. See the nested exception for details."
                ) from ex

    # Update the configuration with `--config`.
    if config_overrides:
        for overrides in config_overrides:
            try:
                unstructured_config = merge_unstructured(unstructured_config, overrides)
            except StructureError as ex:
                raise StructureError(
                    "The command line configuration overrides cannot be merged with the preset recipe configuration. See the nested exception for details."
                ) from ex

    return unstructured_config


class ConfigFileNotFoundError(Exception):
    config_file: Path

    def __init__(self, config_file: Path) -> None:
        super().__init__(
            f"The '{config_file}' path does not point to a configuration file."
        )

        self.config_file = config_file
