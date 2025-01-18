# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from argparse import OPTIONAL, ArgumentParser, BooleanOptionalAction, Namespace
from collections.abc import Hashable, Set
from pathlib import Path
from typing import TypeVar, final

from typing_extensions import override

from fairseq2.cli import CliCommandHandler, setup_logging
from fairseq2.cli.utils.argparse import ConfigAction
from fairseq2.config_registry import ConfigNotFoundError
from fairseq2.context import RuntimeContext
from fairseq2.error import SetupError
from fairseq2.gang import is_torchrun
from fairseq2.logging import log
from fairseq2.recipes.cluster import (
    ClusterError,
    ClusterHandler,
    ClusterResolver,
    UnknownClusterError,
)
from fairseq2.recipes.logging import DistributedLoggingInitializer
from fairseq2.recipes.runner import (
    ConfigFileNotFoundError,
    ConfigReader,
    EnvironmentBootstrapper,
    InferredDefaultDeviceAccessor,
    Recipe,
    RecipeLoader,
    RecipeRunner,
    StandardConfigReader,
    StandardEnvironmentBootstrapper,
    StandardRecipeRunner,
    SystemSignalHandler,
    get_sweep_keys,
)
from fairseq2.recipes.utils.rich import get_console
from fairseq2.recipes.utils.sweep_tagger import (
    NoopSweepTagger,
    StandardSweepTagger,
    SweepFormatError,
    SweepFormatPlaceholderError,
    SweepTagger,
)
from fairseq2.typing import safe_cast
from fairseq2.utils.structured import StructureError, unstructure
from fairseq2.utils.yaml import (
    StandardYamlDumper,
    StandardYamlLoader,
    YamlDumper,
    YamlError,
)

ConfigT = TypeVar("ConfigT")


@final
class RecipeCommandHandler(CliCommandHandler):
    """Runs a recipe over command line."""

    _loader: RecipeLoader[object]
    _config_kls: type[object]
    _default_preset: str
    _extra_sweep_keys: Set[Hashable] | None

    def __init__(
        self,
        loader: RecipeLoader[ConfigT],
        config_kls: type[ConfigT],
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

        def type_erased_loader(
            context: RuntimeContext, config: object, output_dir: Path
        ) -> Recipe:
            config = safe_cast("config", config, config_kls)

            return loader(context, config, output_dir)

        self._loader = type_erased_loader
        self._config_kls = config_kls
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
    def run(
        self, context: RuntimeContext, parser: ArgumentParser, args: Namespace
    ) -> int:
        if args.list_presets:
            self._print_presets(context)

            return 0

        setup_logging(debug=args.debug)

        program = self._create_recipe_program(context, args)

        try:
            program.run(context, args)

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

    def _print_presets(self, context: RuntimeContext) -> None:
        console = get_console()

        recipe_configs = context.get_config_registry(self._config_kls)

        names = recipe_configs.names()

        if names:
            console.print("available presets:")

            for preset in names:
                if preset == self._default_preset:
                    console.print(f"  - {preset} (default)")
                else:
                    console.print(f"  - {preset}")
        else:
            console.print("no preset configuration found.")

    def _create_recipe_program(
        self, context: RuntimeContext, args: Namespace
    ) -> RecipeProgram:
        file_system = context.file_system

        recipe_configs = context.get_config_registry(self._config_kls)

        yaml_loader = StandardYamlLoader(file_system)

        config_reader = StandardConfigReader(recipe_configs, file_system, yaml_loader)

        cluster_handlers = context.get_registry(ClusterHandler)

        cluster_resolver = ClusterResolver(cluster_handlers, is_torchrun=is_torchrun())

        if not args.no_sweep_dir:
            sweep_keys = get_sweep_keys(self._extra_sweep_keys)

            sweep_tagger: SweepTagger = StandardSweepTagger(sweep_keys)
        else:
            sweep_tagger = NoopSweepTagger()

        logging_initializer = DistributedLoggingInitializer(file_system)

        yaml_dumper = StandardYamlDumper(file_system)

        env_bootstrapper = StandardEnvironmentBootstrapper(
            cluster_resolver,
            sweep_tagger,
            file_system,
            logging_initializer,
            yaml_dumper,
        )

        default_device_accessor = InferredDefaultDeviceAccessor()

        signal_handler = SystemSignalHandler()

        runner = StandardRecipeRunner(
            self._loader, default_device_accessor, signal_handler
        )

        return RecipeProgram(config_reader, env_bootstrapper, runner, yaml_dumper)


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

    def run(self, context: RuntimeContext, args: Namespace) -> None:
        config = self._config_reader.read(
            args.preset, args.config_files, args.config_overrides
        )

        if args.dump_config:
            unstructured_config = unstructure(config)

            try:
                self._yaml_dumper.dump(unstructured_config, sys.stdout)
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
        )

        self._runner.run(context, config, output_dir)


class MissingOutputDirectoryError(ValueError):
    pass
