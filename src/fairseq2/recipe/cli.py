# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from argparse import OPTIONAL, ArgumentParser, Namespace
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from signal import SIG_DFL, SIGINT, raise_signal, signal
from typing import Any, TypeVar, final

import torch
from torch.cuda import OutOfMemoryError

from fairseq2.assets import AssetCardError, AssetMetadataError
from fairseq2.checkpoint import (
    CheckpointLoadError,
    CheckpointNotFoundError,
    CheckpointSaveError,
)
from fairseq2.cluster import ClusterError, UnknownClusterError
from fairseq2.data.tokenizers import (
    TokenizerLoadError,
    UnknownTokenizerError,
    UnknownTokenizerFamilyError,
)
from fairseq2.datasets import (
    DataReadError,
    UnknownDatasetError,
    UnknownDatasetFamilyError,
    UnknownSplitError,
)
from fairseq2.error import ContractError, InfraError, InternalError
from fairseq2.file_system import FileSystem
from fairseq2.gang import GangTopologyError, HybridShardingTopologyError
from fairseq2.generation import SequenceGenerationError
from fairseq2.logging import log
from fairseq2.metrics.text import UnknownBleuTokenizerError
from fairseq2.models import (
    ModelLoadError,
    UnknownModelArchitectureError,
    UnknownModelError,
    UnknownModelFamilyError,
)
from fairseq2.nn.data_parallel import DataParallelError
from fairseq2.recipe.base import EvalRecipe, GenerationRecipe, TrainRecipe
from fairseq2.recipe.cluster import WorldInfo
from fairseq2.recipe.compile import TorchCompileError
from fairseq2.recipe.component import UnknownComponentError
from fairseq2.recipe.composition import (
    _register_eval_recipe,
    _register_generation_recipe,
    _register_train_recipe,
)
from fairseq2.recipe.config import _create_config_structurer, get_recipe_config
from fairseq2.recipe.error import (
    ActivationCheckpointingNotSupportedError,
    DatasetNotFoundError,
    FSDPNotSupportedError,
    HuggingFaceNotSupportedError,
    HybridShardingNotSupportedError,
    ModelInitializationError,
    ModelNotFoundError,
    ModelParallelismNotSupportedError,
    TokenizerNotFoundError,
    TorchCompileNotSupportedError,
    UnknownBeamSearchAlgorithmError,
    UnknownLRSchedulerError,
    UnknownMetricDescriptorError,
    UnknownOptimizerError,
    UnknownSamplerError,
    UnknownSequenceGeneratorError,
    UnspecifiedNumberOfStepsError,
)
from fairseq2.recipe.metric_recorders import WandbError
from fairseq2.recipe.run import _run_recipe
from fairseq2.recipe.utils.argparse import ConfigAction
from fairseq2.recipe.utils.sweep_tag import (
    SweepFormatError,
    SweepFormatPlaceholderError,
    SweepTagGenerator,
)
from fairseq2.runtime.composition import ExtensionError, _register_library
from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyNotFoundError,
    DependencyResolver,
    StandardDependencyContainer,
)
from fairseq2.task import TaskStopException
from fairseq2.trainer import InconsistentGradNormError, MinimumLossScaleReachedError
from fairseq2.utils.env import InvalidEnvironmentVariableError
from fairseq2.utils.merge import MergeError, merge_object, to_mergeable
from fairseq2.utils.rich import configure_rich_logging
from fairseq2.utils.structured import StructureError, ValueConverter
from fairseq2.utils.validation import ObjectValidator, ValidationError
from fairseq2.utils.yaml import YamlDumper, YamlError, YamlLoader


def train_main(recipe: TrainRecipe) -> None:
    args = _parse_args()

    configure_rich_logging()

    container = StandardDependencyContainer()

    with _handle_errors(container):
        # Library
        _register_library(container)

        # Recipe
        _register_train_recipe(container, recipe)

        # Recipe Configuration
        def load_config(resolver: DependencyResolver) -> object:
            return _load_config_from_args(resolver, recipe.config_kls)

        container.register(object, load_config, key="config")

        # Recipe Output Directory
        container.register(Path, _create_output_dir, key="output_dir")

        # CLI Arguments
        container.register_instance(Namespace, args)

        # CLI Errors
        _register_cli_errors(container)

        _main(container)


def eval_main(recipe: EvalRecipe) -> None:
    args = _parse_args()

    configure_rich_logging()

    container = StandardDependencyContainer()

    with _handle_errors(container):
        # Library
        _register_library(container)

        # Recipe
        _register_eval_recipe(container, recipe)

        # Recipe Configuration
        def load_config(resolver: DependencyResolver) -> object:
            return _load_config_from_args(resolver, recipe.config_kls)

        container.register(object, load_config, key="config")

        # Recipe Output Directory
        container.register(Path, _create_output_dir, key="output_dir")

        # CLI Arguments
        container.register_instance(Namespace, args)

        # CLI Errors
        _register_cli_errors(container)

        _main(container)


def generate_main(recipe: GenerationRecipe) -> None:
    args = _parse_args()

    configure_rich_logging()

    container = StandardDependencyContainer()

    with _handle_errors(container):
        # Library
        _register_library(container)

        # Recipe
        _register_generation_recipe(container, recipe)

        # Recipe Configuration
        def load_config(resolver: DependencyResolver) -> object:
            return _load_config_from_args(resolver, recipe.config_kls)

        container.register(object, load_config, key="config")

        # Recipe Output Directory
        container.register(Path, _create_output_dir, key="output_dir")

        # CLI Arguments
        container.register_instance(Namespace, args)

        # CLI Errors
        _register_cli_errors(container)

        _main(container)


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


# x AssetCardError
# x AssetMetadataError
# x CheckpointLoadError
# x CheckpointSaveError
# x CheckpointNotFoundError
# x ClusterError
# x DataParallelError
# x DataReadError
# x DatasetNotFoundError
# x FSDPNotSupportedError
# x GangTopologyError
# x HuggingFaceNotSupportedError
# x HybridShardingNotSupportedError
# x HybridShardingTopologyError
# x InvalidEnvironmentVariableError
# x ModelInitializationError?
# x TorchCompileNotSupportedError
# x ModelLoadError
# x ModelNotFoundError
# x ModelParallelismNotSupportedError
# x SequenceGenerationError
# x SweepFormatError
# x SweepFormatPlaceholderError
# x TokenizerLoadError
# x TokenizerNotFoundError
# x TorchCompileError
# x UnknownBeamSearchAlgorithmError
# x UnknownBleuTokenizerError
# x UnknownClusterError
# x UnknownDatasetError
# x UnknownDatasetFamilyError
# x UnknownLRSchedulerError
# x UnknownModelArchitectureError
# x UnknownModelError
# x UnknownModelFamilyError
# x UnknownOptimizerError
# x UnknownSamplerError
# x UnknownSequenceGeneratorError
# x UnspecifiedNumberOfStepsError
# x ValidationError
# x WandbError


# AssetDownloadError?
# DatasetLoadError?


@contextmanager
def _handle_errors(resolver: DependencyResolver) -> Iterator[None]:
    try:
        yield
    except TaskStopException:
        pass
    except StructureError:
        log.exception(
            "Recipe configuration cannot be parsed. See logged stack trace for details."
        )
    except ArgumentError as ex:
        log.error(str(ex), ex=ex.__cause__)

        sys.exit(2)
    except ExtensionError as ex:
        log.exception("{} extension failed to initialize. See logged stack trace for details.", ex.entry_point)  # fmt: skip

        sys.exit(1)
    except InternalError:
        log.exception("Recipe failed with an unexpected internal error. Please file a bug report.")  # fmt: skip

        sys.exit(1)
    except ContractError:
        log.exception("Recipe failed with an unexpected internal error caused by an extension. Please file a bug report to the corresponding extension author.")  # fmt: skip

        sys.exit(1)
    except OutOfMemoryError:
        if log.is_enabled_for_error():
            s = torch.cuda.memory_summary()

            log.exception("CUDA out of memory. See logged memory stats.\n{}", s)

        sys.exit(1)
    except KeyboardInterrupt:
        signal(SIGINT, SIG_DFL)

        raise_signal(SIGINT)
    except Exception as ex:
        handler: Callable[[Any], int] | None

        try:
            handler = resolver.resolve(Callable, key=type(ex))  # type: ignore[arg-type]
        except DependencyNotFoundError:
            handler = None

        if handler is None:
            log.exception("Recipe failed with an unexpected error. See logged stack trace for details.")  # fmt: skip

            ret_code = 1
        else:
            ret_code = handler(ex)

        sys.exit(ret_code)


def _main(resolver: DependencyResolver) -> None:
    args = resolver.resolve(Namespace)

    if args.dump_config:
        _dump_config(resolver)

        return

    if not args.output_dir:
        raise ArgumentError("output_dir", "required")

    _run_recipe(resolver)


def _dump_config(resolver: DependencyResolver) -> None:
    yaml_dumper = resolver.resolve(YamlDumper)

    value_converter = resolver.resolve(ValueConverter)

    recipe_config = get_recipe_config(resolver)

    unstructured_config = value_converter.unstructure(recipe_config)

    if isinstance(unstructured_config, dict):
        unstructured_config = to_mergeable(unstructured_config)

    try:
        yaml_dumper.dump(unstructured_config, sys.stdout)
    except OSError as ex:
        raise InfraError(
            "A system error has occurred while dumping the recipe configuration to `stream`. See the nested exception for details."
        ) from ex


def _load_config_from_args(
    resolver: DependencyResolver, config_kls: type[object]
) -> object:
    args = resolver.resolve(Namespace)

    file_system = resolver.resolve(FileSystem)

    yaml_loader = resolver.resolve(YamlLoader)

    value_converter = resolver.resolve(ValueConverter)

    config_structurer = _create_config_structurer(resolver)

    validator = resolver.resolve(ObjectValidator)

    config_reader = _ConfigReader(config_kls, file_system, yaml_loader, value_converter)

    if args.config_override_files:
        config_override_files = chain.from_iterable(args.config_override_files)
    else:
        config_override_files = None

    unstructured_config = config_reader.read(
        config_override_files, args.config_overrides
    )

    config = config_structurer.structure(unstructured_config, config_kls)

    validator.validate(config)

    return config


@final
class _ConfigReader:
    _kls: type[object]
    _file_system: FileSystem
    _yaml_loader: YamlLoader
    _value_converter: ValueConverter

    def __init__(
        self,
        kls: type[object],
        file_system: FileSystem,
        yaml_loader: YamlLoader,
        value_converter: ValueConverter,
    ) -> None:
        self._kls = kls
        self._file_system = file_system
        self._yaml_loader = yaml_loader
        self._value_converter = value_converter

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
            unstructured_config = self._value_converter.unstructure(config)
        except StructureError as ex:
            raise ContractError(
                "The default recipe configuration cannot be unstructured. See the nested exception for details."
            ) from ex

        # Update the configuration with `--config-override-file`.
        if config_override_files is not None:
            for config_override_file in config_override_files:
                try:
                    is_file = self._file_system.is_file(config_override_file)
                except OSError as ex:
                    raise InfraError(
                        f"A system error has occurred while accessing the '{config_override_file}' configuration file. See the nested exception for details."
                    ) from ex

                if not is_file:
                    raise ArgumentError(
                        "--config-file", f"{config_override_file} does not point to a configuration file."  # fmt: skip
                    ) from None

                try:
                    unstructured_config_overrides = self._yaml_loader.load(
                        config_override_file
                    )
                except YamlError as ex:
                    raise ArgumentError(
                        "--config-file", f"{config_override_file} does not contain a valid recipe configuration override. See logged stack trace for details."  # fmt: skip
                    ) from ex
                except OSError as ex:
                    raise InfraError(
                        f"A system error has occurred while loading the '{config_override_file}' configuration file. See the nested exception for details."
                    ) from ex

                if len(unstructured_config_overrides) == 0:
                    raise ArgumentError(
                        "--config-file", f"{config_override_file} does not contain a recipe configuration."  # fmt: skip
                    )

                try:
                    unstructured_config = merge_object(
                        unstructured_config, unstructured_config_overrides[0]
                    )
                except MergeError as ex:
                    raise ArgumentError(
                        "--config-file", f"{config_override_file} cannot be merged with the recipe configuration. See logged stack trace for details."  # fmt: skip
                    ) from ex

        # Update the configuration with `--config`.
        if config_overrides is not None:
            for overrides in config_overrides:
                try:
                    unstructured_config = merge_object(unstructured_config, overrides)
                except MergeError as ex:
                    raise ArgumentError(
                        "--config", "key-value pair(s) cannot be applied over the recipe configuration. See logged stack trace for details."  # fmt: skip
                    ) from ex

        return unstructured_config


def _create_output_dir(resolver: DependencyResolver) -> Path:
    args = resolver.resolve(Namespace)

    file_system = resolver.resolve(FileSystem)

    output_dir = args.output_dir

    if not args.no_sweep_dir:
        tag = _create_sweep_tag(resolver)

        output_dir = output_dir.joinpath(tag)

    try:
        file_system.make_directory(output_dir)
    except OSError as ex:
        raise InfraError(
            f"A system error has occurred while creating the '{output_dir}' directory. See the nested exception for details."
        ) from ex

    try:
        output_dir = file_system.resolve(output_dir)
    except OSError as ex:
        raise InfraError(
            f"A system error has occurred while resolving the '{output_dir}' directory. See the nested exception for details."
        ) from ex

    return output_dir


def _create_sweep_tag(resolver: DependencyResolver) -> str:
    args = resolver.resolve(Namespace)

    value_converter = resolver.resolve(ValueConverter)

    world_info = resolver.resolve(WorldInfo)

    recipe_config = get_recipe_config(resolver)

    tag_generator = SweepTagGenerator(value_converter)

    # TODO: move to config!
    return tag_generator.generate(recipe_config, world_info.size, args.sweep_format)


class ArgumentError(Exception):
    param_name: str | None

    def __init__(self, param_name: str | None, message: str) -> None:
        if param_name is not None:
            message = f"argument: {param_name}: {message}"

        super().__init__(message)

        self.param_name = param_name


ExceptionT = TypeVar("ExceptionT", bound=Exception)


def register_cli_error(
    container: DependencyContainer,
    kls: type[ExceptionT],
    handler: Callable[[ExceptionT], int],
) -> None:
    container.register_instance(Callable, handler, key=kls)  # type: ignore[arg-type]


def _register_cli_errors(container: DependencyContainer) -> None:
    def register(kls: type[ExceptionT], handler: Callable[[ExceptionT], int]) -> None:
        register_cli_error(container, kls, handler)

    register(ActivationCheckpointingNotSupportedError, _handle_ac_ns_error)
    register(AssetCardError, _handle_asset_card_error)
    register(AssetMetadataError, _handle_asset_metadata_error)
    register(CheckpointLoadError, _handle_checkpoint_load_error)
    register(CheckpointSaveError, _handle_checkpoint_save_error)
    register(CheckpointNotFoundError, _handle_checkpoint_not_found_error)
    register(ClusterError, _handle_cluster_error)
    register(DataParallelError, _handle_data_parallel_error)
    register(DataReadError, _handle_data_read_error)
    register(DatasetNotFoundError, _handle_dataset_not_found_error)
    register(FSDPNotSupportedError, _handle_fsdp_ns_error)
    register(GangTopologyError, _handle_gang_topology_error)
    register(HuggingFaceNotSupportedError, _handle_hg_ns_error)
    register(HybridShardingNotSupportedError, _handle_hsdp_ns_error)
    register(HybridShardingTopologyError, _handle_hsdp_topology_error)
    register(InconsistentGradNormError, _handle_inconsistent_grad_norm_error)
    register(InvalidEnvironmentVariableError, _handle_invalid_env_variable_error)
    register(MinimumLossScaleReachedError, _handle_minimim_loss_scale_reached_error)
    register(ModelInitializationError, _handle_model_init_error)
    register(ModelLoadError, _handle_model_load_error)
    register(ModelNotFoundError, _handle_model_not_found_error)
    register(ModelParallelismNotSupportedError, _handle_model_parallelism_ns_error)
    register(SequenceGenerationError, _handle_seq_generation_error)
    register(SweepFormatError, _handle_sweep_format_error)
    register(SweepFormatPlaceholderError, _handle_sweep_format_placehoder_error)
    register(TokenizerLoadError, _handle_tokenizer_load_error)
    register(TokenizerNotFoundError, _handle_tokenizer_not_found_error)
    register(TorchCompileError, _handle_torch_compile_error)
    register(TorchCompileNotSupportedError, _handle_torch_compile_ns_error)
    register(UnknownBeamSearchAlgorithmError, _handle_unknown_bs_algo_error)
    register(UnknownBleuTokenizerError, _handle_unknown_bleu_tokenizer_error)
    register(UnknownClusterError, _handle_unknown_cluster_error)
    register(UnknownComponentError, _handle_unknown_component_error)
    register(UnknownDatasetError, _handle_unknown_dataset_error)
    register(UnknownDatasetFamilyError, _handle_unknown_dataset_family_error)
    register(UnknownLRSchedulerError, _handle_unknown_lr_scheduler_error)
    register(UnknownMetricDescriptorError, _handle_unknown_metric_desc_error)
    register(UnknownModelArchitectureError, _handle_unknown_model_arch_error)
    register(UnknownModelError, _handle_unknown_model_error)
    register(UnknownModelFamilyError, _handle_unknown_model_family_error)
    register(UnknownOptimizerError, _handle_unknown_optimizer_error)
    register(UnknownSamplerError, _handle_unknown_sampler_error)
    register(UnknownSequenceGeneratorError, _handle_unknown_seq_gen_error)
    register(UnknownSplitError, _handle_unknown_split_error)
    register(UnknownTokenizerError, _handle_unknown_tokenizer_error)
    register(UnknownTokenizerFamilyError, _handle_unknown_tokenizer_family_error)
    register(UnspecifiedNumberOfStepsError, _handle_unspecified_nr_of_steps_error)
    register(ValidationError, _handle_validation_error)
    register(WandbError, _handle_wandb_error)


def _handle_ac_ns_error(ex: ActivationCheckpointingNotSupportedError) -> int:
    log.error("'{}' model does not support activation checkpointing.", ex.model_name)  # fmt: skip

    return 2


def _handle_asset_card_error(ex: AssetCardError) -> int:
    log.exception("'{}' asset card cannot be read. See logged stack trace for details.", ex.name)  # fmt: skip

    return 1


def _handle_asset_metadata_error(ex: AssetMetadataError) -> int:
    log.exception("Asset metadata in '{}' cannot be loaded. See logged stack trace for details.", ex.source)  # fmt: skip

    return 1


def _handle_checkpoint_load_error(ex: CheckpointLoadError) -> int:
    log.exception("Checkpoint of step {} cannot be loaded. See logged stack trace for details.", ex.step_nr)  # fmt: skip

    return 1


def _handle_checkpoint_save_error(ex: CheckpointSaveError) -> int:
    log.exception("Checkpoint of step {} cannot be saved. See logged stack trace for details.", ex.step_nr)  # fmt: skip

    return 1


def _handle_checkpoint_not_found_error(ex: CheckpointNotFoundError) -> int:
    log.error("No checkpoint found for step {}.", ex.step_nr)

    return 2


def _handle_cluster_error(ex: ClusterError) -> int:
    if ex.cluster == "slurm":
        log.exception("'{}' cluster environment cannot be set. See logged stack trace for details. If you are within an allocated Slurm job (i.e. `salloc`), make sure to run with `srun`. If you want to run without Slurm, use `--config common.cluster=none`.", ex.cluster)  # fmt: skip
    else:
        log.exception("'{}' cluster environment cannot be set. See logged stack trace for details.", ex.cluster)  # fmt: skip

    return 1


def _handle_data_parallel_error(ex: DataParallelError) -> int:
    log.exception("Data parallel model cannot be initialized. See logged stack trace for details.")  # fmt: skip

    return 1


def _handle_data_read_error(ex: DataReadError) -> int:
    log.exception("'{}' split of '{}' dataset cannot be read. See logged stack trace for details.", ex.split, ex.dataset_name)  # fmt: skip

    return 1


def _handle_dataset_not_found_error(ex: DatasetNotFoundError) -> int:
    log.error("'{}' path does not point to a dataset.", ex.path)

    return 2


def _handle_fsdp_ns_error(ex: FSDPNotSupportedError) -> int:
    log.error("'{}' model does not support FSDP.", ex.model_name)

    return 2


def _handle_gang_topology_error(ex: GangTopologyError) -> int:
    log.error("`gang.tensor_parallel_size` must be a factor of the number of processes in the root gang ({}), but is {} instead.", ex.root_size, ex.tp_size)  # fmt: skip

    return 2


def _handle_hg_ns_error(ex: HuggingFaceNotSupportedError) -> int:
    log.error("'{}' model does not support conversion to Hugging Face format.", ex.model_name)  # fmt: skip

    return 2


def _handle_hsdp_ns_error(ex: HybridShardingNotSupportedError) -> int:
    log.error("`trainer.fsdp.hybrid` cannot be set when model parallelism is enabled.")

    return 2


def _handle_hsdp_topology_error(ex: HybridShardingTopologyError) -> int:
    log.error("Local world size must be a factor of the number of processes in the data parallel gang ({}) when `trainer.fsdp.hybrid` is set, but is {} instead.", ex.dp_size, ex.intra_node_size)  # fmt: skip

    return 2


def _handle_inconsistent_grad_norm_error(ex: InconsistentGradNormError) -> int:
    log.error("Gradients are inconsistent between processes at step {}. Training cannot continue.", ex.step_nr)  # fmt: skip

    return 3


def _handle_invalid_env_variable_error(ex: InvalidEnvironmentVariableError) -> int:
    log.exception("'{}' environment variable cannot be read. See logged stack trace for details.", ex.name)  # fmt: skip

    return 1


def _handle_minimim_loss_scale_reached_error(ex: MinimumLossScaleReachedError) -> int:
    log.error("Gradients are scaled down to minimum at step {}. Training cannot continue.", ex.step_nr)  # fmt: skip

    return 3


def _handle_model_init_error(ex: ModelInitializationError) -> int:
    log.exception("'{}' model cannot be initialized. See logged stack trace for details.", ex.model_name)  # fmt: skip

    return 1


def _handle_model_load_error(ex: ModelLoadError) -> int:
    log.exception("'{}' model cannot be loaded. See logged stack trace for details.", ex.model_name)  # fmt: skip

    return 1


def _handle_model_not_found_error(ex: ModelNotFoundError) -> int:
    log.error("'{}' path does not point to a model.", ex.path)

    return 2


def _handle_model_parallelism_ns_error(ex: ModelParallelismNotSupportedError) -> int:
    log.error("'{}' model does not support model parallelism.", ex.model_name)

    return 2


def _handle_seq_generation_error(ex: SequenceGenerationError) -> int:
    log.exception("Sequence generation failed. See logged stack trace for details.")

    return 1


def _handle_sweep_format_error(ex: SweepFormatError) -> int:
    log.error(
        "`common.sweep_format` must be a non-empty string with brace-enclosed placeholders."
    )

    return 2


def _handle_sweep_format_placehoder_error(ex: SweepFormatPlaceholderError) -> int:
    s = ", ".join(ex.unknown_keys)

    log.error("`common.sweep_format` must contain only placeholders that correspond to the configuration keys, but contains the following unexpected placeholder(s): {}", s)  # fmt: skip

    return 2


def _handle_tokenizer_load_error(ex: TokenizerLoadError) -> int:
    log.exception("'{}' tokenizer cannot be loaded. See logged stack trace for details.", ex.tokenizer_name)  # fmt: skip

    return 1


def _handle_tokenizer_not_found_error(ex: TokenizerNotFoundError) -> int:
    log.exception("'{}' path does not point to a tokenizer.", ex.path)

    return 2


def _handle_torch_compile_error(ex: TorchCompileError) -> int:
    log.exception("`torch.compile()` call failed. See logged stack trace for details.")

    return 1


def _handle_torch_compile_ns_error(ex: TorchCompileNotSupportedError) -> int:
    log.error("'{}' model does not support `torch.compile()`.", ex.model_name)

    return 2


def _handle_unknown_bs_algo_error(ex: UnknownBeamSearchAlgorithmError) -> int:
    log.error("'{}' is not a known beam search algorithm.", ex.name)

    return 2


def _handle_unknown_bleu_tokenizer_error(ex: UnknownBleuTokenizerError) -> int:
    log.error("'{}' is not a known BLEU tokenizer. See the documentation of the sacrebleu package.")  # fmt: skip

    return 2


def _handle_unknown_cluster_error(ex: UnknownClusterError) -> int:
    log.error("'{}' is not a known cluster.", ex.cluster)

    return 2


def _handle_unknown_component_error(ex: UnknownComponentError) -> int:
    log.error("'{}' is not a known `{}`.", ex.name, ex.kls)

    return 2


def _handle_unknown_dataset_error(ex: UnknownDatasetError) -> int:
    log.error("'{}' is not a known dataset.", ex.dataset_name)

    return 2


def _handle_unknown_dataset_family_error(ex: UnknownDatasetFamilyError) -> int:
    log.error("'{}' is not a known dataset family.", ex.family)

    return 2


def _handle_unknown_lr_scheduler_error(ex: UnknownLRSchedulerError) -> int:
    log.error("'{}' is not a known learning rate scheduler.", ex.name)

    return 2


def _handle_unknown_metric_desc_error(ex: UnknownMetricDescriptorError) -> int:
    log.error("'{}' is not a known metric descriptor.", ex.name)

    return 2


def _handle_unknown_model_arch_error(ex: UnknownModelArchitectureError) -> int:
    log.error("'{}' is not a known architecture of the '{}' model family.", ex.arch, ex.family)  # fmt: skip

    return 2


def _handle_unknown_model_error(ex: UnknownModelError) -> int:
    log.error("'{}' is not a known model.", ex.model_name)

    return 2


def _handle_unknown_model_family_error(ex: UnknownModelFamilyError) -> int:
    log.error("'{}' is not a known model family.", ex.family)

    return 2


def _handle_unknown_optimizer_error(ex: UnknownOptimizerError) -> int:
    log.error("'{}' is not a known optimizer.", ex.name)

    return 2


def _handle_unknown_sampler_error(ex: UnknownSamplerError) -> int:
    log.error("'{}' is not a known sampler.", ex.name)

    return 2


def _handle_unknown_seq_gen_error(ex: UnknownSequenceGeneratorError) -> int:
    log.error("'{}' is not a known sequence generator.", ex.name)

    return 2


def _handle_unknown_split_error(ex: UnknownSplitError) -> int:
    s = ", ".join(sorted(ex.available_splits))

    log.error("'{}' is not a known split of the '{}' dataset. The following splits are available: {}", ex.split, ex.dataset_name, s)  # fmt: skip

    return 2


def _handle_unknown_tokenizer_error(ex: UnknownTokenizerError) -> int:
    log.error("'{}' is not a known tokenizer.", ex.tokenizer_name)

    return 2


def _handle_unknown_tokenizer_family_error(ex: UnknownTokenizerFamilyError) -> int:
    log.error("'{}' is not a known tokenizer family.", ex.family)

    return 2


def _handle_unspecified_nr_of_steps_error(ex: UnspecifiedNumberOfStepsError) -> int:
    log.error("`regime.num_steps` must be specified for the '{}' learning rate scheduler.", ex.lr_scheduler_name)  # fmt: skip

    return 2


def _handle_validation_error(ex: ValidationError) -> int:
    log.error(str(ex.result))

    return 2


def _handle_wandb_error(ex: WandbError) -> int:
    log.exception("Weights & Biases client cannot be initialized. See logged stack trace for details.")  # fmt: skip

    return 1
