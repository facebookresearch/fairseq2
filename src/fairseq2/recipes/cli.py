# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from argparse import (
    OPTIONAL,
    ArgumentError,
    ArgumentParser,
    BooleanOptionalAction,
    Namespace,
)
from collections.abc import Callable, Hashable, Set
from functools import cache
from pathlib import Path
from signal import SIGUSR1, signal
from types import FrameType
from typing import Generic, Protocol, TypeVar, cast, final, runtime_checkable

import yaml
from rich.console import Console
from typing_extensions import override

from fairseq2.config_registry import ConfigRegistry
from fairseq2.error import AlreadyExistsError, InvalidOperationError
from fairseq2.gang import get_world_size
from fairseq2.logging import log
from fairseq2.recipes.console import get_console, set_console
from fairseq2.recipes.logging import setup_basic_logging, setup_logging
from fairseq2.recipes.utils.argparse import ConfigAction
from fairseq2.recipes.utils.environment import (
    EnvironmentSetterRegistry,
    default_env_setters,
)
from fairseq2.recipes.utils.log import log_config
from fairseq2.recipes.utils.sweep_tagger import SweepTagger
from fairseq2.typing import DataClass
from fairseq2.utils.structured import (
    StructureError,
    ValueConverter,
    default_value_converter,
    merge_unstructured,
)


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

    def run(self) -> None:
        """Run the program."""
        set_console(Console(highlight=False))

        parser = ArgumentParser(self._name, description=self._description)

        self.init_parser(parser)

        args = parser.parse_args()

        if not hasattr(args, "command"):
            parser.error("no command specified")

        args.command(args)

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

    def __call__(self, args: Namespace) -> None:
        """Run the command."""
        if self._parser is None:
            raise InvalidOperationError("`init_parser()` must be called first.")

        try:
            self._handler(args)
        except ArgumentError as ex:
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
    def __call__(self, args: Namespace) -> None:
        """Run the command."""


@runtime_checkable
class Stoppable(Protocol):
    """Represents a task that supports graceful stopping."""

    def request_stop(self) -> None:
        ...


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
    _extra_sweep_keys: Set[Hashable] | None
    _value_converter: ValueConverter

    def __init__(
        self,
        loader: RecipeLoader[RecipeConfigT],
        preset_configs: ConfigRegistry[RecipeConfigT],
        default_preset: str,
        *,
        env_setters: EnvironmentSetterRegistry | None = None,
        extra_sweep_keys: Set[Hashable] | None = None,
        value_converter: ValueConverter | None = None,
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
        """
        self._loader = loader
        self._preset_configs = preset_configs
        self._default_preset = default_preset
        self._env_setters = env_setters or default_env_setters
        self._extra_sweep_keys = extra_sweep_keys
        self._value_converter = value_converter or default_value_converter

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

        try:
            unstructured_config = self._value_converter.unstructure(preset_config)
        except StructureError:
            log.exception("Preset configuration '{}' cannot be used. Please file a bug report to the recipe author.", args.preset)  # fmt: skip

            sys.exit(1)

        # Update the configuration with `--config-file`.
        if args.config_files:
            for config_file in args.config_files:
                try:
                    with config_file.open() as fp:
                        unstructured_config_overrides = yaml.safe_load(fp)
                except Exception:
                    log.exception("Configuration file '{}' cannot be read.", config_file)  # fmt: skip

                    sys.exit(1)

                try:
                    unstructured_config = merge_unstructured(
                        unstructured_config, unstructured_config_overrides
                    )
                except StructureError:
                    log.exception("Configuration file '{}' cannot be used.", config_file)  # fmt: skip

                    sys.exit(1)

        # Update the configuration with `--config`.
        if args.config_overrides:
            try:
                unstructured_config = merge_unstructured(
                    unstructured_config, args.config_overrides
                )
            except StructureError:
                log.exception("Command line configuration overrides cannot be applied.")

                sys.exit(1)

        if args.dump_config:
            try:
                yaml.safe_dump(unstructured_config, sys.stdout, sort_keys=False)
            except Exception:
                log.exception("Configuration cannot be dumped to stdout.")

                sys.exit(1)

            sys.exit()

        # If we are not dumping configuration, `--output-dir` is required.
        if not args.output_dir:
            raise ArgumentError(
                None, "the following arguments are required: output_dir"
            )

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
        output_dir = args.output_dir.expanduser().resolve()

        if not args.no_sweep_dir:
            world_size = get_world_size()

            sweep_keys = get_default_sweep_keys()

            if self._extra_sweep_keys is not None:
                sweep_keys = sweep_keys | self._extra_sweep_keys

            sweep_tagger = SweepTagger(world_size, sweep_keys)

            try:
                tag = sweep_tagger.generate(args.preset, unstructured_config)
            except LookupError:
                log.exception("sweep format")

                sys.exit(1)

            output_dir = output_dir.joinpath(tag)

        # Set up distributed logging.
        log_file = output_dir.joinpath("logs/rank_{rank}.log")

        try:
            setup_logging(log_file, debug=args.debug)
        except RuntimeError:
            log.exception("Recipe logging cannot be set up.")

            sys.exit(1)

        log.info("Log files stored under {}.", log_file.parent)

        log_config(unstructured_config, log)

        # Save the configuration to a YAML file.
        config_file = output_dir.joinpath("config.yaml")

        try:
            with config_file.open("w") as fp:
                yaml.safe_dump(unstructured_config, fp, sort_keys=False)
        except Exception:
            log.exception("The configuration cannot be saved to file.")

            sys.exit(1)

        # Parse the configuration.
        try:
            config = self._value_converter.structure(
                unstructured_config, self._preset_configs.config_kls
            )
        except StructureError:
            log.exception("Configuration cannot be parsed.")

            sys.exit(1)

        # Load and run the recipe.
        recipe = self._loader(config, output_dir)

        # If the recipe is stoppable, use SIGUSR1 as the stop signal.
        if isinstance(recipe, Stoppable):

            def request_stop(signum: int, frame: FrameType | None) -> None:
                log.info("SIGUSR1 received. Requesting recipe to stop.")

                cast(Stoppable, recipe).request_stop()

            signal(SIGUSR1, request_stop)

        recipe()


@cache
def get_default_sweep_keys() -> Set[Hashable]:
    return {
        "batch_shuffle_window",
        "betas",
        "data_parallelism",
        "dataset",
        "dtype",
        "example_shuffle_window",
        "final_lr_ratio",
        "final_lr_scale",
        "fp16_loss_scale",
        "fsdp_reshard_after_forward",
        "fsdp_wrap_granularity",
        "gradient_accumulation",
        "label_smoothing",
        "lr",
        "lr_stage_ratios",
        "max_gradient_norm",
        "max_num_elements",
        "max_num_steps",
        "max_num_tokens",
        "max_seq_len",
        "mixed_precision",
        "model",
        "model_arch",
        "model_config",
        "num_lr_warmup_steps",
        "pretrained_model",
        "seed",
        "split",
        "start_lr",
        "start_lr_scale",
        "tensor_parallel_size",
        "tokenizer",
        "train_split",
        "valid_split",
        "weight_decay",
    }
