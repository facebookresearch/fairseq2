# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from argparse import (
    OPTIONAL,
    ArgumentError,
    ArgumentParser,
    BooleanOptionalAction,
    Namespace,
)
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from signal import SIG_DFL, SIGINT, raise_signal, signal
from typing import Any, NoReturn, Protocol, TextIO, TypeVar, final, runtime_checkable

import torch
from torch.cuda import OutOfMemoryError

from fairseq2.assets import (
    AssetCardError,
    AssetDownloadError,
    AssetMetadataError,
    AssetSourceNotFoundError,
)
from fairseq2.checkpoint import CheckpointError, CheckpointNotFoundError
from fairseq2.cluster import ClusterNotDetectedError, ClusterNotKnownError
from fairseq2.composition import ExtensionError, _register_library
from fairseq2.data.tokenizers import (
    TokenizerFamilyNotKnownError,
    TokenizerModelError,
    TokenizerNotKnownError,
)
from fairseq2.datasets import (
    DataReadError,
    DatasetError,
    DatasetFamilyNotKnownError,
    DatasetNotKnownError,
)
from fairseq2.device import LocalRankOutOfRangeError
from fairseq2.error import (
    InternalError,
    OperationalError,
    raise_operational_system_error,
)
from fairseq2.file_system import FileSystem
from fairseq2.generation import SequenceGenerationError
from fairseq2.logging import log
from fairseq2.model_checkpoint import ModelCheckpointError
from fairseq2.models import (
    ModelArchitectureNotKnownError,
    ModelFamilyNotKnownError,
    ModelNotKnownError,
)
from fairseq2.nn.utils.grad import InconsistentGradNormError
from fairseq2.recipe.base import (
    EvalRecipe,
    GenerationRecipe,
    Recipe,
    TrainRecipe,
)
from fairseq2.recipe.component import ComponentNotKnownError
from fairseq2.recipe.config import RecipeConfig
from fairseq2.recipe.error import (
    BeamSearchAlgorithmNotKnownError,
    DeviceTypeNotSupportedError,
    ErrorContext,
    FSDPNotSupportedError,
    GangTopologyError,
    HSDPTopologyError,
    HuggingFaceNotSupportedError,
    LayerwiseACNotSupportedError,
    LRSchedulerNotKnownError,
    ManualGradScalingNotSupportedError,
    MetricNotKnownError,
    MinimumLossScaleReachedError,
    ModelCheckpointNotFoundError,
    ModelTypeNotValidError,
    OptimizerNotKnownError,
    RecipeConfigParseError,
    RecipeError,
    SamplerNotKnownError,
    SequenceGeneratorNotKnownError,
    SplitNotKnownError,
    TokenizerModelNotFoundError,
    TorchCompileError,
    TorchCompileNotSupportedError,
    WandbInitializationError,
)
from fairseq2.recipe.internal.config_preparer import _RecipeConfigPreparer
from fairseq2.recipe.internal.output_dir import _OutputDirectoryCreator
from fairseq2.recipe.run import _run_recipe, _swap_default_resolver
from fairseq2.recipe.task import TaskStopException
from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyNotFoundError,
    DependencyResolver,
)
from fairseq2.utils.argparse import ConfigAction
from fairseq2.utils.config import (
    ConfigDirectiveError,
    ConfigMerger,
    ConfigProcessor,
    ReplacePathDirective,
)
from fairseq2.utils.env import EnvironmentVariableError
from fairseq2.utils.rich import configure_rich_logging
from fairseq2.utils.structured import StructureError, ValueConverter
from fairseq2.utils.validation import ValidationError
from fairseq2.utils.warn import enable_deprecation_warnings
from fairseq2.utils.yaml import YamlDumper, YamlError, YamlLoader


def train_main(recipe: TrainRecipe) -> None:
    from fairseq2.recipe.composition import _register_train_recipe

    enable_deprecation_warnings()

    args = _parse_args()

    configure_rich_logging()

    container = DependencyContainer()

    with _handle_errors(container, args.exit_on_error):
        with _swap_default_resolver(container):
            _register_library(container)

            _register_train_recipe(container, recipe)

            _register_main(container, args, recipe)

            _main(container, args)


@torch.inference_mode()
def eval_main(recipe: EvalRecipe) -> None:
    from fairseq2.recipe.composition import _register_eval_recipe

    enable_deprecation_warnings()

    args = _parse_args()

    configure_rich_logging()

    container = DependencyContainer()

    with _handle_errors(container, args.exit_on_error):
        with _swap_default_resolver(container):
            _register_library(container)

            _register_eval_recipe(container, recipe)

            _register_main(container, args, recipe)

            _main(container, args)


@torch.inference_mode()
def generate_main(recipe: GenerationRecipe) -> None:
    from fairseq2.recipe.composition import _register_generation_recipe

    enable_deprecation_warnings()

    args = _parse_args()

    configure_rich_logging()

    container = DependencyContainer()

    with _handle_errors(container, args.exit_on_error):
        with _swap_default_resolver(container):
            _register_library(container)

            _register_generation_recipe(container, recipe)

            _register_main(container, args, recipe)

            _main(container, args)


def _main(resolver: DependencyResolver, args: Namespace) -> None:
    if args.dump_config:
        printer = resolver.resolve(_RecipeConfigPrinter)

        printer.print(sys.stdout)

        return

    if not args.output_dir:
        raise InternalError("`args.output_dir` is `None`.")

    _run_recipe(resolver)


def _register_main(
    container: DependencyContainer, args: Namespace, recipe: Recipe
) -> None:
    config_kls = recipe.config_kls

    # Recipe Configuration
    def load_config(resolver: DependencyResolver) -> object:
        config_loader = resolver.resolve(_RecipeConfigLoader)

        unstructured_config = config_loader.load(
            config_kls, args.config_file, args.config_overrides
        )

        config_preparer = resolver.resolve(_RecipeConfigPreparer)

        return config_preparer.prepare(config_kls, unstructured_config)

    container.register(RecipeConfig, load_config)

    container.register_type(_RecipeConfigLoader)
    container.register_type(_RecipeConfigPrinter)

    # Recipe Output Directory
    def create_output_dir(resolver: DependencyResolver) -> Path:
        dir_creator = resolver.resolve(_OutputDirectoryCreator)

        return dir_creator.create(args.output_dir)

    container.register(Path, create_output_dir)

    # CLI Errors
    _register_cli_errors(container)


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
        "--exit-on-error",
        default=True,
        action=BooleanOptionalAction,
        help="whether to gracefully exit in case of an error",
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


@contextmanager
def _handle_errors(resolver: DependencyResolver, exit_on_error: bool) -> Iterator[None]:
    def maybe_exit(status: int) -> NoReturn:
        if exit_on_error:
            sys.exit(status)

        raise

    try:
        yield
    except TaskStopException:
        pass
    except RecipeConfigParseError:
        log.exception("Recipe configuration cannot be parsed. See logged stack trace for details.")  # fmt: skip

        maybe_exit(2)
    except ValidationError as ex:
        log.error(str(ex))

        maybe_exit(2)
    except RecipeError as ex:
        msg = str(ex)

        if ex.__cause__ is None:
            log.error(msg)
        else:
            log.exception(msg)

        maybe_exit(2)
    except OperationalError:
        log.exception("Recipe failed due to an operational error. See logged stack trace for details.")  # fmt: skip

        maybe_exit(1)
    except ExtensionError as ex:
        log.exception("{} extension failed to initialize. See logged stack trace for details.", ex.entry_point)  # fmt: skip

        maybe_exit(1)
    except OutOfMemoryError:
        s = torch.cuda.memory_summary()

        log.exception("CUDA out of memory. See logged memory stats.\n{}", s)  # fmt: skip

        maybe_exit(1)
    except KeyboardInterrupt:
        if exit_on_error:
            signal(SIGINT, SIG_DFL)

            raise_signal(SIGINT)

        raise
    except Exception as ex:
        handler: ExceptionHandler[Any] | None

        try:
            handler = resolver.resolve(ExceptionHandler, key=type(ex))  # type: ignore[arg-type]
        except DependencyNotFoundError:
            handler = None

        if handler is None:
            log.exception("Recipe failed due to an unexpected error. See logged stack trace for details and file a bug report to the corresponding author.")  # fmt: skip

            ret_code = 1
        else:
            ret_code = handler(ex)

        maybe_exit(ret_code)


@final
class _RecipeConfigPrinter:
    def __init__(
        self,
        config: RecipeConfig,
        value_converter: ValueConverter,
        yaml_dumper: YamlDumper,
    ) -> None:
        self._config = config
        self._value_converter = value_converter
        self._yaml_dumper = yaml_dumper

    def print(self, stream: TextIO) -> None:
        untyped_config = self._config.as_(object)

        unstructured_config = self._value_converter.unstructure(untyped_config)

        try:
            self._yaml_dumper.dump(unstructured_config, stream)
        except OSError as ex:
            raise_operational_system_error(ex)


@final
class _RecipeConfigLoader:
    def __init__(
        self,
        file_system: FileSystem,
        yaml_loader: YamlLoader,
        value_converter: ValueConverter,
        config_merger: ConfigMerger,
        config_processor: ConfigProcessor,
    ) -> None:
        self._file_system = file_system
        self._yaml_loader = yaml_loader
        self._value_converter = value_converter
        self._config_merger = config_merger
        self._config_processor = config_processor

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
            raise InternalError(
                "Default recipe configuration cannot be unstructured."
            ) from ex

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
                raise RecipeConfigParseError(
                    f"A directive in {config_file} file cannot be processed."
                ) from ex

        if config_overrides is not None:
            for overrides in config_overrides:
                try:
                    unstructured_config = self._config_merger.merge(
                        unstructured_config, overrides
                    )
                except (ValueError, TypeError) as ex:
                    raise RecipeConfigParseError(
                        "Config overrides cannot be applied to the recipe configuration."
                    ) from ex

        return unstructured_config

    def _load_file(self, config_file: Path, unstructured_config: object) -> object:
        try:
            is_file = self._file_system.is_file(config_file)
        except OSError as ex:
            raise_operational_system_error(ex)

        if not is_file:
            raise RecipeConfigParseError(
                f"{config_file} does not point to a configuration file."
            )

        try:
            config_overrides = self._yaml_loader.load(config_file)
        except YamlError as ex:
            raise RecipeConfigParseError(
                f"{config_file} is not a valid YAML file."
            ) from ex
        except OSError as ex:
            raise_operational_system_error(ex)

        if len(config_overrides) == 0:
            raise RecipeConfigParseError(f"{config_file} is empty.")

        try:
            return self._config_merger.merge(unstructured_config, config_overrides[0])
        except (ValueError, TypeError) as ex:
            raise RecipeConfigParseError(
                f"{config_file} cannot be merged with the recipe configuration."
            ) from ex


ExceptionT_contra = TypeVar("ExceptionT_contra", bound=Exception, contravariant=True)


@runtime_checkable
class ExceptionHandler(Protocol[ExceptionT_contra]):
    def __call__(self, ex: ExceptionT_contra) -> int: ...


ExceptionT = TypeVar("ExceptionT", bound=Exception)


def register_cli_error(
    container: DependencyContainer,
    kls: type[ExceptionT],
    handler: ExceptionHandler[ExceptionT],
) -> None:
    container.register_instance(ExceptionHandler, handler, key=kls)  # type: ignore[arg-type]


# fmt: off
def _register_cli_errors(container: DependencyContainer) -> None:
    def register(kls: type[ExceptionT], handler: ExceptionHandler[ExceptionT]) -> None:
        register_cli_error(container, kls, handler)

    register(AssetCardError, _handle_asset_card_error)
    register(AssetDownloadError, _handle_asset_download_error)
    register(AssetMetadataError, _handle_asset_metadata_error)
    register(AssetSourceNotFoundError, _handle_asset_source_not_found_error)
    register(BeamSearchAlgorithmNotKnownError, _handle_bs_algo_not_known_error)
    register(CheckpointError, _handle_checkpoint_error)
    register(CheckpointNotFoundError, _handle_checkpoint_not_found_error)
    register(ClusterNotDetectedError, _handle_cluster_not_detected_error)
    register(ClusterNotKnownError, _handle_cluster_not_known_error)
    register(ComponentNotKnownError, _handle_component_not_known_error)
    register(DataReadError, _handle_data_read_error)
    register(DatasetFamilyNotKnownError, _handle_dataset_family_not_known_error)
    register(DatasetNotKnownError, _handle_dataset_not_known_error)
    register(DatasetError, _handle_dataset_error)
    register(DeviceTypeNotSupportedError, _handle_device_type_not_supported_error)
    register(EnvironmentVariableError, _handle_env_variable_error)
    register(FSDPNotSupportedError, _handle_fsdp_not_supported_error)
    register(GangTopologyError, _handle_gang_topology_error)
    register(HSDPTopologyError, _handle_hsdp_topology_error)
    register(HuggingFaceNotSupportedError, _handle_hg_not_supported_error)
    register(InconsistentGradNormError, _handle_inconsistent_grad_norm_error)
    register(LayerwiseACNotSupportedError, _handle_layerwise_ac_not_supported_error)
    register(LRSchedulerNotKnownError, _handle_lr_scheduler_not_known_error)
    register(LocalRankOutOfRangeError, _handle_local_rank_out_of_range_error)
    register(ManualGradScalingNotSupportedError, _handle_mgs_not_supported_error)
    register(MetricNotKnownError, _handle_metric_not_known_error)
    register(MinimumLossScaleReachedError, _handle_minimim_loss_scale_reached_error)
    register(ModelArchitectureNotKnownError, _handle_model_arch_not_known_error)
    register(ModelCheckpointError, _handle_model_checkpoint_error)
    register(ModelCheckpointNotFoundError, _handle_model_checkpoint_not_found_error)
    register(ModelFamilyNotKnownError, _handle_model_family_not_known_error)
    register(ModelNotKnownError, _handle_model_not_known_error)
    register(ModelTypeNotValidError, _handle_model_type_not_valid_error)
    register(OptimizerNotKnownError, _handle_optimizer_not_known_error)
    register(SamplerNotKnownError, _handle_sampler_not_known_error)
    register(SequenceGenerationError, _handle_seq_generation_error)
    register(SequenceGeneratorNotKnownError, _handle_seq_generator_not_known_error)
    register(SplitNotKnownError, _handle_split_not_known_error)
    register(TokenizerFamilyNotKnownError, _handle_tokenizer_family_not_known_error)
    register(TokenizerModelError, _handle_tokenizer_model_error)
    register(TokenizerModelNotFoundError, _handle_tokenizer_model_not_found_error)
    register(TokenizerNotKnownError, _handle_tokenizer_not_known_error)
    register(TorchCompileError, _handle_torch_compile_error)
    register(TorchCompileNotSupportedError, _handle_torch_compile_not_supported_error)
    register(WandbInitializationError, _handle_wandb_init_error)


def _handle_asset_card_error(ex: AssetCardError) -> int:
    log.exception("{} asset card is erroneous. See logged stack trace for details.", ex.name)

    return 1


def _handle_asset_download_error(ex: AssetDownloadError) -> int:
    log.exception("Failed to download {} {}. See logged stack trace for details.", ex.asset_name, ex.asset_kind)

    return 1


def _handle_asset_metadata_error(ex: AssetMetadataError) -> int:
    log.exception("Asset metadata in {} is erroneous. See logged stack trace for details.", ex.source)

    return 1


def _handle_asset_source_not_found_error(ex: AssetSourceNotFoundError) -> int:
    log.error("{} asset source is not found.", ex.source)

    return 1


def _handle_bs_algo_not_known_error(ex: BeamSearchAlgorithmNotKnownError) -> int:
    log.error("{} is not a known beam search algorithm.", ex.name)

    return 2


def _handle_checkpoint_error(ex: CheckpointError) -> int:
    log.exception("Checkpoint of training step {} is erroneous. See logged stack trace for details.", ex.step_nr)

    return 1


def _handle_checkpoint_not_found_error(ex: CheckpointNotFoundError) -> int:
    log.error("Checkpoint of training step {} is not found.", ex.step_nr)

    return 2


def _handle_cluster_not_detected_error(ex: ClusterNotDetectedError) -> int:
    if ex.cluster == "slurm":
        log.error("{} cluster not detected. If you are within an allocated job (i.e. salloc), make sure to run with srun. If you want to run locally (e.g. via torchrun), use `--config common.cluster=none`.", ex.cluster)
    else:
        log.error("{} cluster not detected.", ex.cluster)

    return 2


def _handle_cluster_not_known_error(ex: ClusterNotKnownError) -> int:
    s = ", ".join(sorted(ex.known_clusters))

    log.error("{} is not a known cluster. `common.cluster` must be one of auto, none, {}.", ex.cluster, s)

    return 2


def _handle_component_not_known_error(ex: ComponentNotKnownError) -> int:
    log.error("{} is not a known `{}`.", ex.name, ex.component_kls)

    return 2


def _handle_data_read_error(ex: DataReadError) -> int:
    log.exception("Failed to read data. See logged stack trace for details.")

    return 1


def _handle_dataset_family_not_known_error(ex: DatasetFamilyNotKnownError) -> int:
    log.error("{} is not a known dataset family.", ex.name)

    return 2


def _handle_dataset_not_known_error(ex: DatasetNotKnownError) -> int:
    log.error("{} is not a known dataset. To see the list of available datasets run: `python -m fairseq2.assets list --kind dataset`.", ex.name)

    return 2


def _handle_dataset_error(ex: DatasetError) -> int:
    section_name = ErrorContext.maybe_get_config_section_name(ex)

    if section_name is None:
        log.exception("Failed to open the dataset. See logged stack trace for details.")
    else:
        log.exception("Failed to open the dataset specified in `{}` section. See logged stack trace for details.", section_name)

    return 1


def _handle_device_type_not_supported_error(ex: DeviceTypeNotSupportedError) -> int:
    log.error("For distributed jobs, only `cpu` and `cuda` devices are supported, but the device of the process is `{}`.", ex.device)

    return 2


def _handle_env_variable_error(ex: EnvironmentVariableError) -> int:
    log.exception("{} environment variable is erroneous. See logged stack trace for details.", ex.var_name)

    return 1


def _handle_fsdp_not_supported_error(ex: FSDPNotSupportedError) -> int:
    log.error("Model does not support FSDP.")

    return 2


def _handle_gang_topology_error(ex: GangTopologyError) -> int:
    log.error("`gang.tensor_parallel_size` must be a factor of the number of processes in the root gang ({}), but is {} instead.", ex.world_size, ex.tp_size)

    return 2


def _handle_hsdp_topology_error(ex: HSDPTopologyError) -> int:
    log.error("Local world size must be a factor of the number of processes in the data parallel gang ({}) when `trainer.fsdp.hybrid` is set, but is {} instead.", ex.dp_size, ex.local_world_size)

    return 2


def _handle_hg_not_supported_error(ex: HuggingFaceNotSupportedError) -> int:
    section_name = ErrorContext.maybe_get_config_section_name(ex)

    if section_name is None:
        log.error("Model does not support exporting to Hugging Face.")
    else:
        log.error("Model specified in `{}` section does not support exporting to Hugging Face.", section_name)

    return 2


def _handle_inconsistent_grad_norm_error(ex: InconsistentGradNormError) -> int:
    s = "\n".join(f"Rank {r:3d} = {g:.8f}" for r, g in enumerate(ex.grad_norms))

    log.error("Gradients are inconsistent between processes at step {}. Training cannot continue. Gradient Norms:\n{}", ex.step_nr, s)

    return 3


def _handle_layerwise_ac_not_supported_error(ex: LayerwiseACNotSupportedError) -> int:
    section_name = ErrorContext.maybe_get_config_section_name(ex)

    if section_name is None:
        log.error("Model does not support layerwise activation checkpointing.")
    else:
        log.error("Model specified in `{}` section does not support layerwise activation checkpointing.", section_name)

    return 2


def _handle_lr_scheduler_not_known_error(ex: LRSchedulerNotKnownError) -> int:
    log.error("{} is not a known learning rate scheduler.", ex.name)

    return 2


def _handle_local_rank_out_of_range_error(ex: LocalRankOutOfRangeError) -> int:
    log.error("Failed to detect the default device of the process. Host has {} {} device(s), but the local rank of the process is {}.", ex.num_devices, ex.device_type, ex.local_rank)

    return 1


def _handle_mgs_not_supported_error(ex: ManualGradScalingNotSupportedError) -> int:
    log.error("Selected optimizer configuration does not support manual fp16 gradient scaling required for FSDP.")

    return 2


def _handle_metric_not_known_error(ex: MetricNotKnownError) -> int:
    log.error("{} is not a known metric.", ex.name)

    return 2


def _handle_minimim_loss_scale_reached_error(ex: MinimumLossScaleReachedError) -> int:
    log.error("Loss is scaled down to minimum at step {}. Training cannot continue.", ex.step_nr)

    return 3


def _handle_model_arch_not_known_error(ex: ModelArchitectureNotKnownError) -> int:
    if ex.family is None:
        log.error("{} is not a known model architecture.", ex.arch)
    else:
        log.error("{} is not a known {} model architecture.", ex.arch, ex.family)

    return 2


def _handle_model_checkpoint_error(ex: ModelCheckpointError) -> int:
    log.exception("Model checkpoint at {} is erroneous. See logged stack trace for details.", ex.path)

    return 1


def _handle_model_checkpoint_not_found_error(ex: ModelCheckpointNotFoundError) -> int:
    log.error("{} does not point to a model checkpoint.", ex.path)

    return 2


def _handle_model_family_not_known_error(ex: ModelFamilyNotKnownError) -> int:
    log.error("{} is not a known model family.", ex.name)

    return 2


def _handle_model_not_known_error(ex: ModelNotKnownError) -> int:
    log.error("{} is not a known model. To see the list of available models run: `python -m fairseq2.assets list --kind model`.", ex.name)

    return 2


def _handle_model_type_not_valid_error(ex: ModelTypeNotValidError) -> int:
    section_name = ErrorContext.maybe_get_config_section_name(ex)

    if section_name is None:
        log.error("Model must be of type `{}`, but is of type `{}` instead.", ex.expected_kls, ex.kls)
    else:
        log.error("Model specified in `{}` section must be of type `{}`, but is of type `{}` instead.", section_name, ex.expected_kls, ex.kls)

    return 2


def _handle_optimizer_not_known_error(ex: OptimizerNotKnownError) -> int:
    log.error("{} is not a known optimizer", ex.name)

    return 2


def _handle_sampler_not_known_error(ex: SamplerNotKnownError) -> int:
    log.error("{} is not a known sampler.", ex.name)

    return 2


def _handle_seq_generation_error(ex: SequenceGenerationError) -> int:
    log.exception("Sequence generation failed. See logged stack trace for details.")

    return 3


def _handle_seq_generator_not_known_error(ex: SequenceGeneratorNotKnownError) -> int:
    log.error("{} is not a known sequence generator.", ex.name)

    return 2


def _handle_split_not_known_error(ex: SplitNotKnownError) -> int:
    section_name = ErrorContext.maybe_get_config_section_name(ex)

    s = ", ".join(sorted(ex.available_splits))

    if section_name is None:
        log.error("{} is not a known dataset split. Available splits are {}.", ex.split, s)
    else:
        log.error("{} specified in `{}` section is not a known dataset split. Available splits are {}.", section_name, ex.split, s)

    return 2


def _handle_tokenizer_family_not_known_error(ex: TokenizerFamilyNotKnownError) -> int:
    log.error("{} is not a known tokenizer family.", ex.name)

    return 2


def _handle_tokenizer_model_error(ex: TokenizerModelError) -> int:
    log.exception("Tokenizer model at {} is erroneous. See logged stack trace for details.", ex.path)

    return 2


def _handle_tokenizer_model_not_found_error(ex: TokenizerModelNotFoundError) -> int:
    log.error("{} does not point to a tokenizer model.", ex.path)

    return 2


def _handle_tokenizer_not_known_error(ex: TokenizerNotKnownError) -> int:
    log.error("{} is not a known tokenizer. To see the list of available tokenizers run: `python -m fairseq2.assets list --kind tokenizer`.", ex.name)

    return 2


def _handle_torch_compile_error(ex: TorchCompileError) -> int:
    section_name = ErrorContext.maybe_get_config_section_name(ex)

    if section_name is None:
        log.exception("`torch.compile()` call failed. See logged stack trace for details.")
    else:
        log.exception("`torch.compile()` call failed for the model specified in `{}` section. See logged stack trace for details.", section_name)

    return 1


def _handle_torch_compile_not_supported_error(ex: TorchCompileNotSupportedError) -> int:
    section_name = ErrorContext.maybe_get_config_section_name(ex)

    if section_name is None:
        log.error("Model does not support torch.compile().")
    else:
        log.error("Model specified in `{}` section does not support torch.compile().", section_name)

    return 2


def _handle_wandb_init_error(ex: WandbInitializationError) -> int:
    log.exception("Weights & Biases client initialization failed. See logged stack trace for details.")

    return 1
# fmt: on
