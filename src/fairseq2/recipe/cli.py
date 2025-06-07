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
from pathlib import Path
from signal import SIG_DFL, SIGINT, raise_signal, signal
from typing import Any, Protocol, TextIO, TypeVar, final, runtime_checkable

import torch
from torch.cuda import OutOfMemoryError

from fairseq2.assets import (
    AssetCardError,
    AssetDownloadError,
    AssetMetadataError,
    AssetSourceNotFoundError,
)
from fairseq2.checkpoint import CheckpointError, CheckpointNotFoundError
from fairseq2.cluster import ClusterNotDetectedError, ClusterNotSupportedError
from fairseq2.composition import ExtensionError, _register_library
from fairseq2.data.tokenizers import (
    TokenizerFamilyNotKnownError,
    TokenizerModelError,
    TokenizerNotKnownError,
)
from fairseq2.datasets import (
    DataReadError,
    DatasetFamilyNotKnownError,
    DatasetNotKnownError,
    DatasetOpenError,
    SplitNotKnownError,
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
from fairseq2.models import ModelFamilyNotKnownError, ModelNotKnownError
from fairseq2.nn.utils.grad import InconsistentGradNormError
from fairseq2.recipe.base import EvalRecipe, GenerationRecipe, TrainRecipe
from fairseq2.recipe.compile import TorchCompileError, TorchCompileNotSupportedError
from fairseq2.recipe.component import ComponentNotKnownError
from fairseq2.recipe.composition import (
    _register_eval_recipe,
    _register_generation_recipe,
    _register_train_recipe,
)
from fairseq2.recipe.config import (
    BeamSearchAlgorithmNotKnownError,
    LRSchedulerNotKnownError,
    OptimizerNotKnownError,
    SamplerNotKnownError,
    SequenceGeneratorNotKnownError,
)
from fairseq2.recipe.config_preparer import RecipeConfigError
from fairseq2.recipe.data_parallel import FSDPNotSupportedError
from fairseq2.recipe.gang import (
    DeviceTypeNotSupportedError,
    GangTopologyError,
    HSDPTopologyError,
)
from fairseq2.recipe.metric_recorders import WandbInitializationError
from fairseq2.recipe.model import (
    ActivationCheckpointingNotSupportedError,
    ModelArchitectureNotKnownError,
    ModelCheckpointNotFoundError,
    ModelParallelismNotSupportedError,
)
from fairseq2.recipe.optim import ManualGradScalingNotSupportedError
from fairseq2.recipe.recipe_preparer import RecipePreparer
from fairseq2.recipe.run import _run_recipe
from fairseq2.recipe.tokenizer import TokenizerModelNotFoundError
from fairseq2.recipe.trainer import HuggingFaceNotSupportedError, MetricNotKnownError
from fairseq2.recipe.wire import _RecipeConfigPreparer
from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyNotFoundError,
    DependencyResolver,
    StandardDependencyContainer,
)
from fairseq2.task import TaskStopException
from fairseq2.trainer import MinimumLossScaleReachedError
from fairseq2.utils.argparse import ConfigAction
from fairseq2.utils.config import ConfigMergeError, ConfigMerger
from fairseq2.utils.env import EnvironmentVariableError
from fairseq2.utils.rich import configure_rich_logging
from fairseq2.utils.structured import StructureError, ValueConverter
from fairseq2.utils.validation import ValidationError
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
            config = _RecipeConfigLoader(resolver).load(
                args.config_file, args.config_overrides
            )

            return _RecipeConfigPreparer(resolver).prepare(config)

        container.register(object, load_config, key="config")

        # Recipe Output Directory
        def create_output_dir(resolver: DependencyResolver) -> Path:
            return resolver.resolve(RecipePreparer).prepare(args.output_dir)

        container.register(Path, create_output_dir)

        # CLI Errors
        _register_cli_errors(container)

        _run(container, args)


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
            config = _RecipeConfigLoader(resolver).load(
                args.config_file, args.config_overrides
            )

            return _RecipeConfigPreparer(resolver).prepare(config)

        container.register(object, load_config, key="config")

        # Recipe Output Directory
        def create_output_dir(resolver: DependencyResolver) -> Path:
            return resolver.resolve(RecipePreparer).prepare(args.output_dir)

        container.register(Path, create_output_dir)

        # CLI Errors
        _register_cli_errors(container)

        _run(container, args)


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
            config = _RecipeConfigLoader(resolver).load(
                args.config_file, args.config_overrides
            )

            return _RecipeConfigPreparer(resolver).prepare(config)

        container.register(object, load_config, key="config")

        # Recipe Output Directory
        def create_output_dir(resolver: DependencyResolver) -> Path:
            return resolver.resolve(RecipePreparer).prepare(args.output_dir)

        container.register(Path, create_output_dir)

        # CLI Errors
        _register_cli_errors(container)

        _run(container, args)


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
        "output_dir",
        type=Path,
        nargs=OPTIONAL,
        help="directory to store recipe artifacts",
    )

    return parser.parse_args()


@contextmanager
def _handle_errors(resolver: DependencyResolver) -> Iterator[None]:
    try:
        yield
    except TaskStopException:
        pass
    except ArgumentError as ex:
        msg = str(ex)

        if ex.__cause__ is None:
            log.error(msg)
        else:
            log.exception(msg)

        sys.exit(2)
    except RecipeConfigError:
        log.exception("Recipe configuration is erroneous. See logged stack trace for details.")  # fmt: skip

        sys.exit(2)
    except ValidationError as ex:
        log.error(str(ex))

        sys.exit(2)
    except ExtensionError as ex:
        log.exception("{} extension failed to initialize. See logged stack trace for details.", ex.entry_point)  # fmt: skip

        sys.exit(1)
    except OperationalError:
        log.exception("Recipe failed due to an operational error. See logged stack trace for details.")  # fmt: skip

        sys.exit(1)
    except InternalError:
        log.exception("Recipe failed due to an unexpected internal error. Please file a bug report.")  # fmt: skip

        sys.exit(1)
    except OutOfMemoryError:
        s = torch.cuda.memory_summary()

        log.exception("CUDA out of memory. See logged memory stats.\n{}", s)

        sys.exit(1)
    except KeyboardInterrupt:
        signal(SIGINT, SIG_DFL)

        raise_signal(SIGINT)
    except Exception as ex:
        handler: ExceptionHandler[Any] | None

        try:
            handler = resolver.resolve(ExceptionHandler, key=type(ex))  # type: ignore[arg-type]
        except DependencyNotFoundError:
            handler = None

        if handler is None:
            log.exception("Recipe failed due to an unexpected error. See logged stack trace for details.")  # fmt: skip

            ret_code = 1
        else:
            ret_code = handler(ex)

        sys.exit(ret_code)


def _run(resolver: DependencyResolver, args: Namespace) -> None:
    if args.dump_config:
        printer = _RecipeConfigPrinter(resolver, sys.stdout)

        printer.print()

        return

    if not args.output_dir:
        raise ArgumentError("output_dir", "required")

    _run_recipe(resolver)


def _RecipeConfigPrinter(
    resolver: DependencyResolver, stream: TextIO
) -> RecipeConfigPrinter:
    config = resolver.resolve(object, key="config")

    yaml_dumper = resolver.resolve(YamlDumper)

    value_converter = resolver.resolve(ValueConverter)

    return RecipeConfigPrinter(config, value_converter, yaml_dumper, stream)


@final
class RecipeConfigPrinter:
    def __init__(
        self,
        config: object,
        value_converter: ValueConverter,
        yaml_dumper: YamlDumper,
        stream: TextIO,
    ) -> None:
        self._config = config
        self._value_converter = value_converter
        self._yaml_dumper = yaml_dumper
        self._stream = stream

    def print(self) -> None:
        unstructured_config = self._value_converter.unstructure(self._config)

        try:
            self._yaml_dumper.dump(unstructured_config, self._stream)
        except OSError as ex:
            raise_operational_system_error(ex)


def _RecipeConfigLoader(resolver: DependencyResolver) -> RecipeConfigLoader:
    config_kls = resolver.resolve(type, key="config_kls")

    file_system = resolver.resolve(FileSystem)

    yaml_loader = resolver.resolve(YamlLoader)

    value_converter = resolver.resolve(ValueConverter)

    config_merger = resolver.resolve(ConfigMerger)

    return RecipeConfigLoader(
        config_kls, file_system, yaml_loader, value_converter, config_merger
    )


@final
class RecipeConfigLoader:
    def __init__(
        self,
        config_kls: type[object],
        file_system: FileSystem,
        yaml_loader: YamlLoader,
        value_converter: ValueConverter,
        config_merger: ConfigMerger,
    ) -> None:
        self._config_kls = config_kls
        self._file_system = file_system
        self._yaml_loader = yaml_loader
        self._value_converter = value_converter
        self._config_merger = config_merger

    def load(
        self,
        config_file: Path | None,
        config_overrides: Iterable[Mapping[str, object]] | None,
    ) -> object:
        try:
            config = self._config_kls()
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
            unstructured_config = self._load_file(config_file, unstructured_config)

        if config_overrides is not None:
            for overrides in config_overrides:
                try:
                    unstructured_config = self._config_merger.merge(
                        unstructured_config, overrides
                    )
                except ConfigMergeError as ex:
                    msg = "key-value pair(s) cannot be applied over the recipe configuration. See logged stack trace for details."

                    raise ArgumentError("--config", msg) from ex

        return unstructured_config

    def _load_file(self, config_file: Path, unstructured_config: object) -> object:
        try:
            is_file = self._file_system.is_file(config_file)
        except OSError as ex:
            raise_operational_system_error(ex)

        if not is_file:
            msg = f"{config_file} does not point to a configuration file."

            raise ArgumentError("--config-file", msg)

        try:
            config_overrides = self._yaml_loader.load(config_file)
        except YamlError as ex:
            msg = f"{config_file} is not a valid YAML file. See logged stack trace for details."

            raise ArgumentError("--config-file", msg) from ex
        except OSError as ex:
            raise_operational_system_error(ex)

        if len(config_overrides) == 0:
            msg = f"{config_file} is empty."

            raise ArgumentError("--config-file", msg)

        try:
            return self._config_merger.merge(unstructured_config, config_overrides[0])
        except ConfigMergeError as ex:
            msg = f"{config_file} cannot be merged with the recipe configuration. See logged stack trace for details."

            raise ArgumentError("--config-file", msg) from ex


class ArgumentError(Exception):
    param_name: str

    def __init__(self, param_name: str, message: str) -> None:
        super().__init__(f"argument: {param_name}: {message}")

        self.param_name = param_name


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


def _register_cli_errors(container: DependencyContainer) -> None:
    def register(kls: type[ExceptionT], handler: ExceptionHandler[ExceptionT]) -> None:
        register_cli_error(container, kls, handler)

    register(ActivationCheckpointingNotSupportedError, _handle_ac_not_supported_error)
    register(AssetCardError, _handle_asset_card_error)
    register(AssetDownloadError, _handle_asset_download_error)
    register(AssetMetadataError, _handle_asset_metadata_error)
    register(AssetSourceNotFoundError, _handle_asset_source_not_found_error)
    register(BeamSearchAlgorithmNotKnownError, _handle_bs_algo_not_known_error)
    register(CheckpointError, _handle_checkpoint_error)
    register(CheckpointNotFoundError, _handle_checkpoint_not_found_error)
    register(ClusterNotDetectedError, _handle_cluster_not_detected_error)
    register(ClusterNotSupportedError, _handle_cluster_not_supported_error)
    register(ComponentNotKnownError, _handle_component_not_known_error)
    register(DataReadError, _handle_data_read_error)
    register(DatasetFamilyNotKnownError, _handle_dataset_family_not_known_error)
    register(DatasetNotKnownError, _handle_dataset_not_known_error)
    register(DatasetOpenError, _handle_dataset_open_error)
    register(DeviceTypeNotSupportedError, _handle_device_type_not_supported_error)
    register(EnvironmentVariableError, _handle_env_variable_error)
    register(FSDPNotSupportedError, _handle_fsdp_not_supported_error)
    register(GangTopologyError, _handle_gang_topology_error)
    register(HSDPTopologyError, _handle_hsdp_topology_error)
    register(HuggingFaceNotSupportedError, _handle_hg_not_supported_error)
    register(InconsistentGradNormError, _handle_inconsistent_grad_norm_error)
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
    register(ModelParallelismNotSupportedError, _handle_mp_not_supported_error)
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


def _handle_ac_not_supported_error(ex: ActivationCheckpointingNotSupportedError) -> int:
    model = "Model" if ex.model_name == "train" else f"{ex.model_name} model"

    log.error("{} does not support activation checkpointing.", model)  # fmt: skip

    return 2


def _handle_asset_card_error(ex: AssetCardError) -> int:
    log.exception("{} asset card is erroneous. See logged stack trace for details.", ex.name)  # fmt: skip

    return 1


def _handle_asset_download_error(ex: AssetDownloadError) -> int:
    log.exception("Failed to download the {} {}. See logged stack trace for details.", ex.asset_name, ex.asset_kind)  # fmt: skip

    return 1


def _handle_asset_metadata_error(ex: AssetMetadataError) -> int:
    log.exception("Asset metadata in {} is erroneous. See logged stack trace for details.", ex.source)  # fmt: skip

    return 1


def _handle_asset_source_not_found_error(ex: AssetSourceNotFoundError) -> int:
    log.error("{} asset source is not found.", ex.source)  # fmt: skip

    return 1


def _handle_bs_algo_not_known_error(ex: BeamSearchAlgorithmNotKnownError) -> int:
    log.error("{} is not a known beam search algorithm.", ex.name)  # fmt: skip

    return 2


def _handle_checkpoint_error(ex: CheckpointError) -> int:
    log.error("Checkpoint of training step {} is erroneous. See logged stack trace for details.", ex.step_nr)  # fmt: skip

    return 1


def _handle_checkpoint_not_found_error(ex: CheckpointNotFoundError) -> int:
    log.error("Checkpoint of training step {} is not found.", ex.step_nr)  # fmt: skip

    return 2


def _handle_cluster_not_detected_error(ex: ClusterNotDetectedError) -> int:
    if ex.cluster == "slurm":
        log.error("{} cluster not detected. If you are within an allocated job (i.e. salloc), make sure to run with srun. If you want to run locally (e.g. via torchrun), use `--config common.cluster=none`.", ex.cluster)  # fmt: skip
    else:
        log.error("{} cluster not detected.", ex.cluster)  # fmt: skip

    return 2


def _handle_cluster_not_supported_error(ex: ClusterNotSupportedError) -> int:
    s = ", ".join(sorted(ex.supported_clusters))

    log.error("{} is not a supported cluster. `common.cluster` must be one of auto, none, {}.", ex.cluster, s)  # fmt: skip

    return 2


def _handle_component_not_known_error(ex: ComponentNotKnownError) -> int:
    log.error("{} is not a known `{}`.", ex.name, ex.component_kls)  # fmt: skip

    return 2


def _handle_data_read_error(ex: DataReadError) -> int:
    dataset = "dataset" if ex.dataset_name == "custom" else f"{ex.dataset_name} dataset"

    log.exception("Failed to read from the {} split of the {}. See logged stack trace for details.", ex.split, dataset)  # fmt: skip

    return 1


def _handle_dataset_family_not_known_error(ex: DatasetFamilyNotKnownError) -> int:
    log.error("{} is not a known dataset family.", ex.family)

    return 2


def _handle_dataset_not_known_error(ex: DatasetNotKnownError) -> int:
    log.error("{} is not a known dataset.", ex.name)  # fmt: skip

    return 2


def _handle_dataset_open_error(ex: DatasetOpenError) -> int:
    dataset = "dataset" if ex.dataset_name == "custom" else f"{ex.dataset_name} dataset"

    log.exception("Failed to open the {}. See logged stack trace for details.", dataset)

    return 1


def _handle_device_type_not_supported_error(ex: DeviceTypeNotSupportedError) -> int:
    log.error("For distributed jobs, only `cpu` and `cuda` devices are supported, but the device of the process is `{}`.", ex.device)  # fmt: skip

    return 2


def _handle_env_variable_error(ex: EnvironmentVariableError) -> int:
    log.exception("{} environment variable is erroneous. See logged stack trace for details.", ex.var_name)  # fmt: skip

    return 1


def _handle_fsdp_not_supported_error(ex: FSDPNotSupportedError) -> int:
    model = "Model" if ex.model_name == "train" else f"{ex.model_name} model"

    log.error("{} does not support FSDP.", model)  # fmt: skip

    return 2


def _handle_gang_topology_error(ex: GangTopologyError) -> int:
    log.error("`gang.tensor_parallel_size` must be a factor of the number of processes in the root gang ({}), but is {} instead.", ex.world_size, ex.tp_size)  # fmt: skip

    return 2


def _handle_hsdp_topology_error(ex: HSDPTopologyError) -> int:
    log.error("Local world size must be a factor of the number of processes in the data parallel gang ({}) when `trainer.fsdp.hybrid` is set, but is {} instead.", ex.dp_size, ex.local_world_size)  # fmt: skip

    return 2


def _handle_hg_not_supported_error(ex: HuggingFaceNotSupportedError) -> int:
    model = "Model" if ex.model_name == "train" else f"{ex.model_name} model"

    log.error("{} does not support Hugging Face conversion.", model)  # fmt: skip

    return 2


def _handle_inconsistent_grad_norm_error(ex: InconsistentGradNormError) -> int:
    s = "\n".join(f"Rank {r:3d} = {g:.8f}" for r, g in enumerate(ex.grad_norms))

    log.error("Gradients are inconsistent between processes at step {}. Training cannot continue. Gradient Norms:\n{}", ex.step_nr, s)  # fmt: skip

    return 3


def _handle_lr_scheduler_not_known_error(ex: LRSchedulerNotKnownError) -> int:
    log.error("{} is not a known learning rate scheduler.", ex.name)  # fmt: skip

    return 2


def _handle_local_rank_out_of_range_error(ex: LocalRankOutOfRangeError) -> int:
    log.error("Failed to detect the default device of the process. Host has {} {} device(s), but the local rank of the process is {}.", ex.num_devices, ex.device_type, ex.local_rank)  # fmt: skip

    return 1


def _handle_mgs_not_supported_error(ex: ManualGradScalingNotSupportedError) -> int:
    log.error("{} optimizer does not support manual fp16 gradient scaling which is required for FSDP.", ex.optimizer_name)  # fmt: skip

    return 2


def _handle_metric_not_known_error(ex: MetricNotKnownError) -> int:
    log.error("{} is not a known metric.", ex.name)  # fmt: skip

    return 2


def _handle_minimim_loss_scale_reached_error(ex: MinimumLossScaleReachedError) -> int:
    log.error("Gradients are scaled down to minimum at step {}. Training cannot continue.", ex.step_nr)  # fmt: skip

    return 3


def _handle_model_arch_not_known_error(ex: ModelArchitectureNotKnownError) -> int:
    log.error("{} is not a known model architecture.", ex.arch)

    return 2


def _handle_model_checkpoint_error(ex: ModelCheckpointError) -> int:
    log.error("Model checkpoint at {} is erroneous. See logged stack trace for details.", ex.path)  # fmt: skip

    return 2


def _handle_model_checkpoint_not_found_error(ex: ModelCheckpointNotFoundError) -> int:
    log.error("{} does not point to a model checkpoint.", ex.path)  # fmt: skip

    return 2


def _handle_model_family_not_known_error(ex: ModelFamilyNotKnownError) -> int:
    log.error("{} is not a known model family.", ex.family)  # fmt: skip

    return 2


def _handle_model_not_known_error(ex: ModelNotKnownError) -> int:
    log.error("{} is not a known model.", ex.name)

    return 2


def _handle_mp_not_supported_error(ex: ModelParallelismNotSupportedError) -> int:
    log.error("{} model does not support model parallelism.", ex.model_name)

    return 2


def _handle_optimizer_not_known_error(ex: OptimizerNotKnownError) -> int:
    log.error("{} is not a known optimizer", ex.name)  # fmt: skip

    return 2


def _handle_sampler_not_known_error(ex: SamplerNotKnownError) -> int:
    log.error("{} is not a known sampler.", ex.name)

    return 2


def _handle_seq_generation_error(ex: SequenceGenerationError) -> int:
    log.exception("Sequence generation failed. See logged stack trace for details.")  # fmt: skip

    return 3


def _handle_seq_generator_not_known_error(ex: SequenceGeneratorNotKnownError) -> int:
    log.error("{} is not a known sequence generator.", ex.name)

    return 2


def _handle_split_not_known_error(ex: SplitNotKnownError) -> int:
    dataset = "dataset" if ex.dataset_name == "custom" else f"{ex.dataset_name} dataset"

    s = ", ".join(sorted(ex.available_splits))

    log.error("{} is not a known split of the {}. Available splits are {}.", ex.split, dataset, s)  # fmt: skip

    return 2


def _handle_tokenizer_family_not_known_error(ex: TokenizerFamilyNotKnownError) -> int:
    log.error("{} is not a known tokenizer family.", ex.family)  # fmt: skip

    return 2


def _handle_tokenizer_model_error(ex: TokenizerModelError) -> int:
    log.error("Tokenizer model at {} is erroneous. See logged stack trace for details.", ex.path)  # fmt: skip

    return 2


def _handle_tokenizer_model_not_found_error(ex: TokenizerModelNotFoundError) -> int:
    log.error("{} does not point to a tokenizer model.", ex.path)  # fmt: skip

    return 2


def _handle_tokenizer_not_known_error(ex: TokenizerNotKnownError) -> int:
    log.error("{} is not a known tokenizer.", ex.name)  # fmt: skip

    return 2


def _handle_torch_compile_error(ex: TorchCompileError) -> int:
    log.exception("`torch.compile()` call failed. See logged stack trace for details.")  # fmt: skip

    return 1


def _handle_torch_compile_not_supported_error(ex: TorchCompileNotSupportedError) -> int:
    model = "Model" if ex.model_name == "train" else f"{ex.model_name} model"

    log.error("{} does not support torch.compile().", model)  # fmt: skip

    return 2


def _handle_wandb_init_error(ex: WandbInitializationError) -> int:
    log.exception("Weights & Biases client initialization failed. See logged stack trace for details.")  # fmt: skip

    return 1
