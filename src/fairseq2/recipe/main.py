# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from argparse import OPTIONAL, ArgumentParser, Namespace
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from functools import partial
from itertools import chain
from pathlib import Path
from signal import SIG_DFL, SIGINT, raise_signal, signal

import torch
from torch.cuda import OutOfMemoryError

from fairseq2.cluster import UnknownClusterError
from fairseq2.composition import ExtensionError, register_library
from fairseq2.data.tokenizers import (
    UnknownTokenizerError,
    UnknownTokenizerFamilyError,
)
from fairseq2.datasets import (
    InvalidDatasetTypeError,
    UnknownDatasetError,
    UnknownDatasetFamilyError,
    UnknownSplitError,
)
from fairseq2.dependency import (
    DependencyContainer,
    DependencyResolver,
    StandardDependencyContainer,
)
from fairseq2.error import ContractError, InternalError, SetupError
from fairseq2.file_system import FileSystem
from fairseq2.logging import log
from fairseq2.metrics.recorders import UnknownMetricDescriptorError
from fairseq2.metrics.text import UnknownBleuTokenizerError
from fairseq2.models import (
    InvalidModelTypeError,
    UnknownModelArchitectureError,
    UnknownModelError,
    UnknownModelFamilyError,
)
from fairseq2.recipe.base import EvalRecipe, GenerationRecipe, TrainRecipe
from fairseq2.recipe.composition import (
    register_eval_recipe,
    register_generation_recipe,
    register_train_recipe,
)
from fairseq2.recipe.config import get_config
from fairseq2.recipe.error import (
    ActivationCheckpointingNotSupportedError,
    DatasetPathNotFoundError,
    FSDPNotSupportedError,
    HuggingFaceNotSupportedError,
    HybridShardingNotSupportedError,
    InconsistentGradNormError,
    MinimumLossScaleReachedError,
    ModelCompilationNotSupportedError,
    ModelParallelismNotSupportedError,
    ModelPathNotFoundError,
    UnspecifiedNumberOfStepsError,
)
from fairseq2.recipe.run import run_recipe
from fairseq2.recipe.task import TaskStopException
from fairseq2.recipe.utils.argparse import ConfigAction
from fairseq2.recipe.utils.sweep_tag import (
    SweepFormatError,
    SweepFormatPlaceholderError,
    SweepTagGenerator,
)
from fairseq2.utils.env import InvalidEnvironmentVariableError, get_env, get_world_size
from fairseq2.utils.merge import MergeError, merge_object, to_mergeable
from fairseq2.utils.structured import StructureError, structure, unstructure
from fairseq2.utils.validation import ValidationError, validate
from fairseq2.utils.yaml import YamlDumper, YamlError, YamlLoader


def train_main(recipe: TrainRecipe) -> None:
    args = _parse_args()

    container = StandardDependencyContainer()

    with error_handler(container):
        # Library
        register_library(container)

        # Recipe
        register_train_recipe(container, recipe)

        # Recipe Configuration
        load_config = partial(_load_config, config_kls=recipe.config_kls)

        container.register(object, load_config, key="config")

        # Recipe Output Directory
        container.register(Path, _make_output_dir, key="output_dir")

        # CLI Arguments
        container.register_instance(Namespace, args)

        # User Errors
        _register_user_error_types(container)

        main(container)


def eval_main(recipe: EvalRecipe) -> None:
    args = _parse_args()

    container = StandardDependencyContainer()

    with error_handler(container):
        # Library
        register_library(container)

        # Recipe
        register_eval_recipe(container, recipe)

        # Recipe Configuration
        load_config = partial(_load_config, config_kls=recipe.config_kls)

        container.register(object, load_config, key="config")

        # Recipe Output Directory
        container.register(Path, _make_output_dir, key="output_dir")

        # CLI Arguments
        container.register_instance(Namespace, args)

        # User Errors
        _register_user_error_types(container)

        main(container)


def generation_main(recipe: GenerationRecipe) -> None:
    args = _parse_args()

    container = StandardDependencyContainer()

    with error_handler(container):
        # Library
        register_library(container)

        # Recipe
        register_generation_recipe(container, recipe)

        # Recipe Configuration
        load_config = partial(_load_config, config_kls=recipe.config_kls)

        container.register(object, load_config, key="config")

        # Recipe Output Directory
        container.register(Path, _make_output_dir, key="output_dir")

        # CLI Arguments
        container.register_instance(Namespace, args)

        # User Errors
        _register_user_error_types(container)

        main(container)


@contextmanager
def error_handler(resolver: DependencyResolver) -> Iterator[None]:
    exit_code = 0

    try:
        yield
    except TaskStopException:
        pass
    except SetupError:
        log.exception("Recipe initialization has failed. See the logged stack trace for details.")  # fmt: skip

        exit_code = 1
    except CliArgumentError as ex:
        log.error(str(ex), ex=ex.__cause__)

        exit_code = 2
    except InternalError:
        log.exception("Task failed with an unexpected internal error. Please file a bug report.")  # fmt: skip

        exit_code = 1
    except ContractError:
        log.exception("Task failed with an unexpected internal error caused by an extension. Please file a bug report to the corresponding extension author.")  # fmt: skip

        exit_code = 1
    except ExtensionError as ex:
        log.exception("{} extension failed to load. See the logged stack trace for details.", ex.entry_point)  # fmt: skip

        exit_code = 1
    except OutOfMemoryError:
        if log.is_enabled_for_error():
            s = torch.cuda.memory_summary()

            log.exception("CUDA out of memory. See logged memory stats.\n{}", s)

        exit_code = 1
    except KeyboardInterrupt:
        signal(SIGINT, SIG_DFL)

        raise_signal(SIGINT)
    except Exception as ex:
        if log.is_enabled_for_error():
            user_error_types = resolver.resolve_all(type, key="user_error")

            if type(ex) in user_error_types:
                log.error(str(ex))
            else:
                log.exception("Task failed with an unexpected error. See the logged stack trace for details.")  # fmt: skip

        exit_code = 1

    if exit_code > 0:
        sys.exit(exit_code)


def main(resolver: DependencyResolver) -> None:
    try:
        args = resolver.resolve(Namespace)

        if args.dump_config:
            _dump_config(resolver)

            return

        if not args.output_dir:
            raise CliArgumentError("output_dir", "required")

        run_recipe(resolver)
    except UnknownClusterError as ex:
        s = ", ".join(ex.supported_clusters)

        raise CliArgumentError(
            "cluster", f"'{ex.cluster}' is not a known cluster. Must be one of: auto, none, {s}"  # fmt: skip
        ) from None
    except ConfigFileNotFoundError as ex:
        raise CliArgumentError(
            "--config-file", f"{ex.config_file} does not point to a configuration file."  # fmt: skip
        ) from None
    except InvalidConfigFileError as ex:
        raise CliArgumentError(
            "--config-file", f"{ex.config_file} does not contain a valid configuration override. See the logged stack trace for details."  # fmt: skip
        ) from ex
    except InvalidConfigOverrideError as ex:
        raise CliArgumentError(
            "--config", "key-value pair(s) cannot be applied over the preset configuration. See the logged stack trace for details."  # fmt: skip
        ) from ex
    except SweepFormatPlaceholderError as ex:
        s = ", ".join(ex.unknown_keys)

        raise CliArgumentError(
            "--sweep-format", f"must contain only placeholders that correspond to the configuration keys, but contains the following unexpected placeholder(s): {s}"  # fmt: skip
        ) from None
    except SweepFormatError:
        raise CliArgumentError(
            "--sweep-format", "must be a non-empty string with brace-enclosed placeholders."  # fmt: skip
        ) from None
    except StructureError as ex:
        raise CliArgumentError(
            None, "The recipe configuration cannot be parsed. See the logged stack trace for details."  # fmt: skip
        ) from ex


def _parse_args() -> Namespace:
    parser = ArgumentParser()

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
        default="ws_{world_size}.{hash}",
        help="format of the sweep directory name (default: %(default)s)",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        nargs=OPTIONAL,
        help="directory to store recipe artifacts",
    )

    return parser.parse_args()


def _load_config(resolver: DependencyResolver, config_kls: type[object]) -> object:
    args = resolver.resolve(Namespace)

    file_system = resolver.resolve(FileSystem)

    yaml_loader = resolver.resolve(YamlLoader)

    config_reader = _ConfigReader(config_kls, file_system, yaml_loader)

    if args.config_override_files:
        config_override_files = chain.from_iterable(args.config_override_files)
    else:
        config_override_files = None

    try:
        config = config_reader.read(config_override_files, args.config_overrides)
    except ConfigFileReadError as ex:
        raise SetupError(
            "The recipe configuration cannot be read. See the nested exception for details."
        ) from ex

    validate(config)

    return config


class _ConfigReader:
    _kls: type[object]
    _file_system: FileSystem
    _yaml_loader: YamlLoader

    def __init__(
        self, kls: type[object], file_system: FileSystem, yaml_loader: YamlLoader
    ) -> None:
        self._kls = kls
        self._file_system = file_system
        self._yaml_loader = yaml_loader

    def read(
        self,
        config_override_files: Iterable[Path] | None,
        config_overrides: Iterable[Mapping[str, object]] | None,
    ) -> object:
        try:
            config = self._kls()
        except TypeError as ex:
            raise ContractError(
                "The default recipe configuration cannot be constructed. See the nested exception for details."
            ) from ex

        try:
            unstructured_config = unstructure(config)
        except StructureError as ex:
            raise ContractError(
                "The recipe configuration cannot be unstructured. See the nested exception for details."
            ) from ex

        # Update the configuration with `--config-override-file`.
        if config_override_files:
            for config_override_file in config_override_files:
                try:
                    is_file = self._file_system.is_file(config_override_file)
                except OSError as ex:
                    raise ConfigFileReadError(
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
                    raise ConfigFileReadError(
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
                        config_override_file, f"The '{config_override_file}' configuration file cannot be merged with the recipe configuration. See the nested exception for details."  # fmt: skip
                    ) from ex

        # Update the configuration with `--config`.
        if config_overrides:
            for overrides in config_overrides:
                try:
                    unstructured_config = merge_object(unstructured_config, overrides)
                except MergeError as ex:
                    raise InvalidConfigOverrideError(
                        "The configuration overrides cannot be merged with the recipe configuration. See the nested exception for details."
                    ) from ex

        return structure(unstructured_config, self._kls)


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


class ConfigFileReadError(Exception):
    pass


def _dump_config(resolver: DependencyResolver) -> None:
    config = get_config(resolver)

    yaml_dumper = resolver.resolve(YamlDumper)

    if isinstance(config, Mapping):
        config = to_mergeable(config)

    try:
        yaml_dumper.dump(config, sys.stdout)
    except YamlError as ex:
        raise SetupError(
            "The recipe configuration cannot be dumped to stdout. See the nested exception for details."
        ) from ex


def _make_output_dir(resolver: DependencyResolver) -> Path:
    args = resolver.resolve(Namespace)

    file_system = resolver.resolve(FileSystem)

    output_dir: Path = args.output_dir

    if not args.no_sweep_dir:
        env = get_env(resolver)

        try:
            world_size = get_world_size(env)
        except InvalidEnvironmentVariableError as ex:
            raise SetupError(
                "The world size cannot be determined. See the nested exception for details."
            ) from ex

        config = get_config(resolver)

        tag_generator = SweepTagGenerator(world_size, args.sweep_format)

        config = unstructure(config)

        tag = tag_generator.generate(config)

        output_dir = output_dir.joinpath(tag)

    try:
        file_system.make_directory(output_dir)
    except OSError as ex:
        raise SetupError(
            f"The '{output_dir}' recipe directory cannot be created. See the nested exception for details."
        ) from ex

    return output_dir


class CliArgumentError(Exception):
    param_name: str | None

    def __init__(self, param_name: str | None, message: str) -> None:
        if param_name is not None:
            message = f"argument: {param_name}: {message}"

        super().__init__(message)

        self.param_name = param_name


def _register_user_error_types(container: DependencyContainer) -> None:
    def register(kls: type[Exception]) -> None:
        container.register_instance(type, kls, key="user_error")

    register(ActivationCheckpointingNotSupportedError)
    register(DatasetPathNotFoundError)
    register(FSDPNotSupportedError)
    register(HuggingFaceNotSupportedError)
    register(HybridShardingNotSupportedError)
    register(InconsistentGradNormError)
    register(InvalidDatasetTypeError)
    register(InvalidModelTypeError)
    register(MinimumLossScaleReachedError)
    register(ModelCompilationNotSupportedError)
    register(ModelParallelismNotSupportedError)
    register(ModelPathNotFoundError)
    register(UnknownBleuTokenizerError)
    register(UnknownDatasetError)
    register(UnknownDatasetFamilyError)
    register(UnknownMetricDescriptorError)
    register(UnknownModelArchitectureError)
    register(UnknownModelError)
    register(UnknownModelFamilyError)
    register(UnknownSplitError)
    register(UnknownTokenizerError)
    register(UnknownTokenizerFamilyError)
    register(UnspecifiedNumberOfStepsError)
    register(ValidationError)
