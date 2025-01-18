# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Set
from functools import cache
from itertools import chain
from pathlib import Path
from signal import SIGUSR1, signal
from types import FrameType
from typing import (
    Mapping,
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
    final,
    runtime_checkable,
)

from typing_extensions import override

from fairseq2.config_registry import ConfigProvider
from fairseq2.context import RuntimeContext
from fairseq2.error import ContractError, SetupError
from fairseq2.logging import log
from fairseq2.recipes.cluster import ClusterResolver
from fairseq2.recipes.logging import LoggingInitializer
from fairseq2.recipes.utils.log import log_config
from fairseq2.recipes.utils.sweep_tagger import SweepTagger
from fairseq2.utils.file import FileSystem
from fairseq2.utils.structured import (
    StructureError,
    merge_unstructured,
    structure,
    unstructure,
)
from fairseq2.utils.yaml import YamlDumper, YamlError, YamlLoader


class RecipeRunner(ABC):
    @abstractmethod
    def run(self, context: RuntimeContext, config: object, output_dir: Path) -> None:
        ...


Recipe: TypeAlias = Callable[[], None]


ConfigT_contra = TypeVar("ConfigT_contra", contravariant=True)


class RecipeLoader(Protocol[ConfigT_contra]):
    def __call__(
        self, context: RuntimeContext, config: ConfigT_contra, output_dir: Path
    ) -> Recipe:
        ...


ConfigT = TypeVar("ConfigT")


@final
class StandardRecipeRunner(RecipeRunner):
    _loader: RecipeLoader[object]
    _signal_handler: SignalHandler

    def __init__(
        self, loader: RecipeLoader[object], signal_handler: SignalHandler
    ) -> None:
        self._loader = loader
        self._signal_handler = signal_handler

    @override
    def run(self, context: RuntimeContext, config: object, output_dir: Path) -> None:
        recipe = self._loader(context, config, output_dir)

        # If the recipe is stoppable, use SIGUSR1 as the stop signal.
        if isinstance(recipe, Stoppable):

            def request_stop(nr: int) -> None:
                log.info("SIGUSR1 received. Requesting recipe to stop.")

                recipe.request_stop()

            self._signal_handler.set(SIGUSR1, request_stop)

        recipe()


@runtime_checkable
class Stoppable(Protocol):
    """Represents a task that supports graceful stopping."""

    def request_stop(self) -> None:
        ...


class SignalHandler(ABC):
    @abstractmethod
    def set(self, nr: int, callback: Callable[[int], None]) -> None:
        ...


@final
class SystemSignalHandler(SignalHandler):
    @override
    def set(self, nr: int, callback: Callable[[int], None]) -> None:
        def cb(signum: int, frame: FrameType | None) -> None:
            callback(signum)

        signal(nr, cb)


class EnvironmentBootstrapper(ABC):
    @abstractmethod
    def run(
        self,
        preset: str,
        config: object,
        output_dir: Path,
        *,
        cluster: str = "auto",
        sweep_format: str | None = None,
    ) -> Path:
        ...


@final
class StandardEnvironmentBootstrapper(EnvironmentBootstrapper):
    _cluster_resolver: ClusterResolver
    _sweep_tagger: SweepTagger
    _file_system: FileSystem
    _logging_initializer: LoggingInitializer
    _yaml_dumper: YamlDumper

    def __init__(
        self,
        cluster_resolver: ClusterResolver,
        sweep_tagger: SweepTagger,
        file_system: FileSystem,
        logging_initializer: LoggingInitializer,
        yaml_dumper: YamlDumper,
    ) -> None:
        self._cluster_resolver = cluster_resolver
        self._sweep_tagger = sweep_tagger
        self._file_system = file_system
        self._logging_initializer = logging_initializer
        self._yaml_dumper = yaml_dumper

    @override
    def run(
        self,
        preset: str,
        config: object,
        output_dir: Path,
        *,
        cluster: str = "auto",
        sweep_format: str | None = None,
    ) -> Path:
        cluster_handler = self._cluster_resolver.get(cluster)

        world_size, rank = cluster_handler.set_torch_distributed_variables()

        unstructured_config = unstructure(config)

        sweep_tag = self._sweep_tagger.generate(
            world_size, preset, unstructured_config, sweep_format
        )

        sweep_output_dir = output_dir.joinpath(sweep_tag)

        try:
            self._file_system.make_directory(sweep_output_dir)
        except OSError as ex:
            raise SetupError(
                f"The '{sweep_output_dir}' recipe output directory cannot be created. See the nested exception for details."
            ) from ex

        self._logging_initializer.initialize(
            sweep_output_dir.joinpath("logs/rank_{rank}.log")
        )

        log.info("The log files stored under the '{}' directory.", sweep_output_dir)

        log_config(unstructured_config, log)

        if rank == 0:
            config_file = sweep_output_dir.joinpath("config.yaml")

            try:
                self._yaml_dumper(unstructured_config, config_file)
            except (OSError, YamlError) as ex:
                raise SetupError(
                    f"The recipe configuration cannot be saved to the '{config_file}' file. See the nested exception for details."
                ) from ex

        return sweep_output_dir


class ConfigReader(ABC):
    @abstractmethod
    def read(
        self,
        preset: str,
        config_files: Sequence[Sequence[Path]] | None,
        config_overrides: Sequence[Mapping[str, object]] | None,
    ) -> object:
        ...


@final
class StandardConfigReader(ConfigReader):
    _preset_configs: ConfigProvider[object]
    _file_system: FileSystem
    _yaml_loader: YamlLoader

    def __init__(
        self,
        preset_configs: ConfigProvider[object],
        file_system: FileSystem,
        yaml_loader: YamlLoader,
    ) -> None:
        self._preset_configs = preset_configs
        self._file_system = file_system
        self._yaml_loader = yaml_loader

    @override
    def read(
        self,
        preset: str,
        config_files: Sequence[Sequence[Path]] | None,
        config_overrides: Sequence[Mapping[str, object]] | None,
    ) -> object:
        # Load the preset configuration.
        preset_config = self._preset_configs.get(preset)

        try:
            unstructured_config = unstructure(preset_config)
        except StructureError as ex:
            raise ContractError(
                f"The '{preset}' preset configuration cannot be unstructured. See the nested exception for details."
            ) from ex

        # Update the configuration with `--config-file`.
        if config_files:
            for config_file in chain.from_iterable(config_files):
                if not self._file_system.is_file(config_file):
                    raise ConfigFileNotFoundError(config_file)

                try:
                    unstructured_config_overrides = self._yaml_loader(config_file)
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
                    unstructured_config = merge_unstructured(
                        unstructured_config, overrides
                    )
                except StructureError as ex:
                    raise StructureError(
                        "The command line configuration overrides cannot be merged with the preset recipe configuration. See the nested exception for details."
                    ) from ex

        return structure(unstructured_config, self._preset_configs.config_kls)  # type: ignore[no-any-return]


class ConfigFileNotFoundError(Exception):
    config_file: Path

    def __init__(self, config_file: Path) -> None:
        super().__init__(
            f"The '{config_file}' path does not point to a configuration file."
        )

        self.config_file = config_file


def get_sweep_keys(extra_sweep_keys: Set[Hashable] | None) -> Set[Hashable]:
    sweep_keys = get_default_sweep_keys()

    if extra_sweep_keys is not None:
        sweep_keys = sweep_keys | extra_sweep_keys

    return sweep_keys


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
