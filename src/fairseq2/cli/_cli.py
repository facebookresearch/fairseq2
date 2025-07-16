# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace

from rich.console import Console

from fairseq2.cli.utils.rich import set_console
from fairseq2.context import RuntimeContext
from fairseq2.error import AlreadyExistsError, InvalidOperationError
from fairseq2.logging import log


class Cli:
    """Represents the entry point of a command line program."""

    _name: str
    _description: str | None
    _origin_module: str
    _version: str
    _groups: dict[str, CliGroup]
    _user_error_types: set[type[Exception]]

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
        self._user_error_types = set()

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

    def run(self, context: RuntimeContext) -> int:
        """Run the program."""
        set_console(Console(highlight=False))

        parser = ArgumentParser(self._name, description=self._description)

        self.init_parser(parser)

        args = parser.parse_args()

        if not hasattr(args, "command"):
            parser.error("no command specified")

        try:
            return args.command.run(context, args)  # type: ignore[no-any-return]
        except CliCommandError:
            log.exception("Command failed. See the logged stack trace for details.")

            return 1
        except CliArgumentError as ex:
            log.error(str(ex), ex=ex.__cause__)

            return 2
        except Exception as ex:
            if type(ex) in self._user_error_types:
                log.error(str(ex))

                return 1

            raise

    def register_user_error_type(self, kls: type[Exception]) -> None:
        self._user_error_types.add(kls)

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

    def run(self, context: RuntimeContext, args: Namespace) -> int:
        """Run the command."""
        if self._parser is None:
            raise InvalidOperationError("`init_parser()` must be called first.")

        try:
            return self._handler.run(context, self._parser, args)
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
    def run(
        self, context: RuntimeContext, parser: ArgumentParser, args: Namespace
    ) -> int:
        """Run the command."""


class CliCommandError(Exception):
    pass


class CliArgumentError(Exception):
    param_name: str | None

    def __init__(self, param_name: str | None, message: str) -> None:
        if param_name is not None:
            message = f"argument: {param_name}: {message}"

        super().__init__(message)

        self.param_name = param_name
