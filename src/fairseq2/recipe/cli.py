# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from argparse import OPTIONAL, ArgumentError, ArgumentParser, Namespace
from collections.abc import Iterator, Mapping
from pathlib import Path
from signal import SIG_DFL, SIGINT, raise_signal, signal
from typing import TextIO, final

import torch
from torch.cuda import OutOfMemoryError

from fairseq2.composition import ExtensionError, _register_library
from fairseq2.error import (
    InternalError,
    OperationalError,
    raise_operational_system_error,
)
from fairseq2.file_system import FileSystem
from fairseq2.logging import configure_logging, log
from fairseq2.recipe.base import Recipe
from fairseq2.recipe.error import ConfigError, RecipeError
from fairseq2.recipe.internal.config import (
    _is_train_config,
    _RecipeConfigHolder,
    _RecipeConfigStructurer,
    _StandardRecipeConfigStructurer,
)
from fairseq2.recipe.internal.output_dir import _OutputDirectoryCreator
from fairseq2.recipe.run import _run_recipe, _swap_default_resolver
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver
from fairseq2.utils.argparse import ConfigAction
from fairseq2.utils.config import (
    ConfigDirectiveError,
    ConfigMerger,
    ConfigProcessor,
    ReplacePathDirective,
)
from fairseq2.utils.env import EnvironmentVariableError
from fairseq2.utils.structured import StructureError, ValueConverter
from fairseq2.utils.validation import ObjectValidator, ValidationError
from fairseq2.utils.warn import _warn_deprecated, enable_deprecation_warnings
from fairseq2.utils.yaml import YamlDumper, YamlError, YamlLoader

#
# DEPRECATED - BEGIN
#


def train_main(recipe: Recipe) -> None:
    enable_deprecation_warnings()

    _warn_deprecated(
        "`train_main()` is deprecated and will be removed in v0.14. Use `main()` instead."
    )

    main(recipe)


def eval_main(recipe: Recipe) -> None:
    enable_deprecation_warnings()

    _warn_deprecated(
        "`eval_main()` is deprecated and will be removed in v0.14. Use `main()` instead."
    )

    main(recipe)


def generate_main(recipe: Recipe) -> None:
    enable_deprecation_warnings()

    _warn_deprecated(
        "`generate_main()` is deprecated and will be removed in v0.14. Use `main()` instead."
    )

    main(recipe)


#
# DEPRECATED - END
#


def main(recipe: Recipe) -> int:
    try:
        _run(recipe)
    except ValidationError as ex:
        log.error(str(ex))

        return 2
    except ConfigError as ex:
        log.error(str(ex), exc=ex.__cause__)

        return 2
    except RecipeError as ex:
        log.error(str(ex), exc=ex.__cause__)

        return 1
    except OperationalError:
        log.exception("Recipe failed due to an operational error.")

        return 1
    except OutOfMemoryError:
        s = torch.cuda.memory_summary()

        log.exception("CUDA out of memory. See logged memory stats.\n{}", s)

        return 1
    except KeyboardInterrupt:
        signal(SIGINT, SIG_DFL)

        raise_signal(SIGINT)

        return 1
    except Exception:
        log.exception("Recipe failed due to an unexpected error. File a bug report to the corresponding author.")  # fmt: skip

        return 1
    else:
        return 0


def _run(recipe: Recipe) -> None:
    from fairseq2.recipe.composition import (
        _register_inference_recipe,
        _register_train_recipe,
    )

    args = _parse_args()

    enable_deprecation_warnings()

    configure_logging(no_rich=args.no_rich)

    is_train_recipe = _is_train_config(recipe.config_kls)

    container = DependencyContainer()

    with _swap_default_resolver(container):
        with torch.inference_mode(mode=not is_train_recipe):
            try:
                _register_library(container, no_progress=True if args.no_rich else None)
            except ExtensionError as ex:
                raise OperationalError(
                    f"{ex.entry_point} extension failed to initialize."
                ) from ex

            if is_train_recipe:
                _register_train_recipe(container, recipe)
            else:
                _register_inference_recipe(container, recipe)

            _register_run(container, args, recipe)

            if args.dump_config:
                printer = container.resolve(_RecipeConfigPrinter)

                printer.print(sys.stdout)

                return

            if not args.output_dir:
                raise InternalError("`args.output_dir` is `None`.")

            _run_recipe(container)


def _parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--config-file",
        dest="config_file",
        metavar="CONFIG_FILE",
        type=Path,
        nargs=OPTIONAL,
        help="configuration file",
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
        "--no-rich",
        default=False,
        action="store_true",
        help="whether to disable rich text output for logging",
    )

    parser.add_argument(
        "--no-exit-on-error",
        default=False,
        action="store_true",
        help="whether to propagate unhandled errors",
    )

    output_dir_action = parser.add_argument(
        "output_dir",
        type=Path,
        nargs=OPTIONAL,
        help="directory to store recipe artifacts",
    )

    args = parser.parse_args()

    if not args.dump_config and not args.output_dir:
        err = ArgumentError(output_dir_action, "required")

        parser.error(str(err))

    return args


def _register_run(
    container: DependencyContainer, args: Namespace, recipe: Recipe
) -> None:
    config_kls = recipe.config_kls

    # Recipe Configuration
    def get_config(resolver: DependencyResolver) -> _RecipeConfigHolder:
        config_loader = resolver.resolve(_RecipeConfigLoader)

        config = config_loader.load(config_kls, args.config_file, args.config_overrides)

        validator = resolver.resolve(ObjectValidator)

        validator.validate(config)

        return _RecipeConfigHolder(config)

    container.register(_RecipeConfigHolder, get_config)

    container.register_type(_RecipeConfigLoader)
    container.register_type(_RecipeConfigStructurer, _StandardRecipeConfigStructurer)

    # Recipe Output Directory
    def get_output_dir(resolver: DependencyResolver) -> Path:
        dir_creator = resolver.resolve(_OutputDirectoryCreator)

        return dir_creator.create(args.output_dir)

    container.register(Path, get_output_dir)

    container.register_type(_RecipeConfigPrinter)


@final
class _RecipeConfigPrinter:
    def __init__(
        self,
        config_holder: _RecipeConfigHolder,
        value_converter: ValueConverter,
        yaml_dumper: YamlDumper,
    ) -> None:
        self._config_holder = config_holder
        self._value_converter = value_converter
        self._yaml_dumper = yaml_dumper

    def print(self, stream: TextIO) -> None:
        try:
            unstructured_config = self._value_converter.unstructure(
                self._config_holder.config
            )
        except StructureError as ex:
            raise InternalError("Failed to unstructure recipe configuration.") from ex

        try:
            self._yaml_dumper.dump(unstructured_config, stream)
        except YamlError as ex:
            raise InternalError(
                "Failed to serialize recipe configuration to YAML."
            ) from ex
        except OSError as ex:
            raise OperationalError(
                "Failed to dump recipe configuration to stdout."
            ) from ex


@final
class _RecipeConfigLoader:
    def __init__(
        self,
        file_system: FileSystem,
        yaml_loader: YamlLoader,
        value_converter: ValueConverter,
        config_merger: ConfigMerger,
        config_processor: ConfigProcessor,
        config_structurer: _RecipeConfigStructurer,
    ) -> None:
        self._file_system = file_system
        self._yaml_loader = yaml_loader
        self._value_converter = value_converter
        self._config_merger = config_merger
        self._config_processor = config_processor
        self._config_structurer = config_structurer

    def load(
        self,
        config_kls: type[object],
        config_file: Path | None,
        config_overrides: Iterator[Mapping[str, object]] | None,
    ) -> object:
        try:
            config = config_kls()
        except TypeError as ex:
            raise InternalError(
                "Default recipe configuration cannot be constructed."
            ) from ex

        try:
            unstructured_config = self._value_converter.unstructure(config)
        except StructureError as ex:
            raise InternalError("Recipe configuration cannot be unstructured.") from ex

        if config_file is not None:
            try:
                config_file = self._file_system.resolve(config_file)
            except OSError as ex:
                raise_operational_system_error(ex)

            unstructured_config = self._load_file(config_file, unstructured_config)

            directive = ReplacePathDirective(config_file.parent)

            try:
                unstructured_config = self._config_processor.process(
                    unstructured_config, [directive]
                )
            except ConfigDirectiveError as ex:
                raise RecipeError(
                    f"A directive in {config_file} file cannot be processed."
                ) from ex

        if config_overrides is not None:
            for overrides in config_overrides:
                try:
                    unstructured_config = self._config_merger.merge(
                        unstructured_config, overrides
                    )
                except TypeError as ex:
                    raise RecipeError(
                        "--config overrides cannot be applied to the recipe configuration."
                    ) from ex

        try:
            return self._config_structurer.structure(config_kls, unstructured_config)
        except StructureError as ex:
            raise RecipeError("Recipe configuration cannot be structured.") from ex

    def _load_file(self, config_file: Path, unstructured_config: object) -> object:
        try:
            is_file = self._file_system.is_file(config_file)
        except OSError as ex:
            raise_operational_system_error(ex)

        if not is_file:
            raise RecipeError(f"{config_file} does not point to a configuration file.")

        try:
            config_overrides = self._yaml_loader.load(config_file)
        except YamlError as ex:
            raise RecipeError(f"{config_file} is not a valid YAML file.") from ex
        except OSError as ex:
            raise_operational_system_error(ex)

        if len(config_overrides) == 0:
            raise RecipeError(f"{config_file} is empty.")

        try:
            return self._config_merger.merge(unstructured_config, config_overrides[0])
        except TypeError as ex:
            raise RecipeError(
                f"{config_file} cannot be merged with the recipe configuration."
            ) from ex


#    register(AssetCardError, _handle_asset_card_error)
#    register(AssetDownloadError, _handle_asset_download_error)
#    register(CorruptAssetMetadataError, _handle_asset_metadata_error)
#    register(AssetSourceNotFoundError, _handle_asset_source_not_found_error)
#    register(CheckpointError, _handle_checkpoint_error)
#    register(CheckpointNotFoundError, _handle_checkpoint_not_found_error)
#    register(DataReadError, _handle_data_read_error)
#    register(DatasetError, _handle_dataset_error)
#    register(DatasetFamilyNotKnownError, _handle_dataset_family_not_known_error)
#    register(DatasetNotKnownError, _handle_dataset_not_known_error)
#    register(InconsistentGradNormError, _handle_inconsistent_grad_norm_error)
#    register(ModelArchitectureNotKnownError, _handle_model_arch_not_known_error)
#    register(CorruptModelCheckpointError, _handle_corrupt_model_checkpoint_error)
#    register(ModelFamilyNotKnownError, _handle_model_family_not_known_error)
#    register(ModelGatedError, _handle_model_gated_error)
#    register(ModelNotKnownError, _handle_model_not_known_error)
#    register(SequenceGenerationError, _handle_seq_generation_error)
#    register(TokenizerFamilyNotKnownError, _handle_tokenizer_family_not_known_error)
#    register(TokenizerGatedError, _handle_tokenizer_gated_error)
#    register(TokenizerModelError, _handle_tokenizer_model_error)
#    register(TokenizerNotKnownError, _handle_tokenizer_not_known_error)
#
#
# def _handle_asset_card_error(ex: AssetCardError) -> int:
#    log.exception("{} asset card is erroneous. See logged stack trace for details.", ex.name)
#
#    return 1
#
#
# def _handle_asset_download_error(ex: AssetDownloadError) -> int:
#    log.exception("Failed to download {}. See logged stack trace for details.", ex.uri)
#
#    return 1
#
#
# def _handle_asset_metadata_error(ex: CorruptAssetMetadataError) -> int:
#    log.exception("Asset metadata in {} is erroneous. See logged stack trace for details.", ex.source)
#
#    return 1
#
#
# def _handle_asset_source_not_found_error(ex: AssetSourceNotFoundError) -> int:
#    log.error("{} asset source is not found.", ex.source)
#
#    return 1
#
#
# def _handle_checkpoint_error(ex: CheckpointError) -> int:
#    log.exception("Checkpoint of training step {} is erroneous. See logged stack trace for details.", ex.step_nr)
#
#    return 1
#
#
# def _handle_checkpoint_not_found_error(ex: CheckpointNotFoundError) -> int:
#    log.error("Checkpoint of training step {} is not found.", ex.step_nr)
#
#    return 2
#
#
# def _handle_data_read_error(ex: DataReadError) -> int:
#    log.exception("Failed to read data. See logged stack trace for details.")
#
#    return 1
#
#
# def _handle_dataset_family_not_known_error(ex: DatasetFamilyNotKnownError) -> int:
#    log.error("{} is not a known dataset family.", ex.name)
#
#    return 2
#
#
# def _handle_dataset_not_known_error(ex: DatasetNotKnownError) -> int:
#    log.error("{} is not a known dataset. To see the list of available datasets run: `python -m fairseq2.assets list --kind dataset`.", ex.name)
#
#    return 2
#
#
# def _handle_dataset_error(ex: DatasetError) -> int:
#    log.exception("Failed to open the dataset. See logged stack trace for details.")
#
#    return 1
#
#
# def _handle_inconsistent_grad_norm_error(ex: InconsistentGradNormError) -> int:
#    s = "\n".join(f"Rank {r:3d} = {g:.8f}" for r, g in enumerate(ex.grad_norms))
#
#    log.error("Gradients are inconsistent between processes at step {}. Training cannot continue. Gradient Norms:\n{}", ex.step_nr, s)
#
#    return 3
#
#
# def _handle_model_arch_not_known_error(ex: ModelArchitectureNotKnownError) -> int:
#    if ex.family is None:
#        log.error("{} is not a known model architecture.", ex.arch)
#    else:
#        log.error("{} is not a known {} model architecture.", ex.arch, ex.family)
#
#    return 2
#
#
# def _handle_corrupt_model_checkpoint_error(ex: CorruptModelCheckpointError) -> int:
#    log.exception("Model checkpoint at {} is erroneous. See logged stack trace for details.", ex.path)
#
#    return 1
#
#
# def _handle_model_family_not_known_error(ex: ModelFamilyNotKnownError) -> int:
#    log.error("{} is not a known model family.", ex.name)
#
#    return 2
#
#
# def _handle_model_gated_error(ex: ModelGatedError) -> int:
#    if ex.info_url:
#        log.error("{} is a gated model. See {} for more information.", ex.name, ex.info_url)
#    else:
#        log.error("{} is a gated model.", ex.name)
#
#    return 2
#
#
# def _handle_model_not_known_error(ex: ModelNotKnownError) -> int:
#    log.error("{} is not a known model. To see the list of available models run: `python -m fairseq2.assets list --kind model`.", ex.name)
#
#    return 2
#
#
# def _handle_seq_generation_error(ex: SequenceGenerationError) -> int:
#    log.exception("Sequence generation failed. See logged stack trace for details.")
#
#    return 3
#
#
# def _handle_tokenizer_family_not_known_error(ex: TokenizerFamilyNotKnownError) -> int:
#    log.error("{} is not a known tokenizer family.", ex.name)
#
#    return 2
#
#
# def _handle_tokenizer_gated_error(ex: TokenizerGatedError) -> int:
#    if ex.info_url:
#        log.error("{} is a gated tokenizer. See {} for more information.", ex.name, ex.info_url)
#    else:
#        log.error("{} is a gated tokenizer.", ex.name)
#
#    return 2
#
#
# def _handle_tokenizer_model_error(ex: TokenizerModelError) -> int:
#    log.exception("Tokenizer model at {} is erroneous. See logged stack trace for details.", ex.path)
#
#    return 2
#
#
# def _handle_tokenizer_not_known_error(ex: TokenizerNotKnownError) -> int:
#    log.error("{} is not a known tokenizer. To see the list of available tokenizers run: `python -m fairseq2.assets list --kind tokenizer`.", ex.name)
#
#    return 2
## fmt: on
