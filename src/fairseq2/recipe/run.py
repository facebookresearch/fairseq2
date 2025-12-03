# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import final

import torch

import fairseq2.runtime.dependency
from fairseq2.assets import (
    AssetMetadataLoadError,
    AssetMetadataSourceNotFoundError,
    AssetStore,
    CorruptAssetMetadataError,
)
from fairseq2.composition import _register_library
from fairseq2.error import InternalError, OperationalError
from fairseq2.logging import configure_logging
from fairseq2.recipe.base import Recipe
from fairseq2.recipe.error import RecipeError, ConfigError
from fairseq2.recipe.internal.cluster import _ClusterPreparer
from fairseq2.recipe.internal.config import _is_train_config, _RecipeConfigHolder
from fairseq2.recipe.internal.log import _LogHelper
from fairseq2.recipe.internal.logging import _DistributedLogConfigurer
from fairseq2.recipe.internal.output_dir import _OutputDirectoryCreator
from fairseq2.recipe.internal.task import _TaskRunner
from fairseq2.recipe.internal.torch import _TorchConfigurer
from fairseq2.recipe.task import Task
from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyResolver,
    activate_dependency,
)
from fairseq2.utils.env import EnvironmentVariableError
from fairseq2.utils.structured import StructureError, ValueConverter
from fairseq2.utils.validation import ObjectValidator
from fairseq2.utils.warn import _warn_deprecated, enable_deprecation_warnings
from fairseq2.utils.yaml import YamlDumper, YamlError
from fairseq2.world_info import WorldInfo

#
# DEPRECATED - BEGIN
#


def train(recipe: Recipe, config: object, output_dir: Path) -> None:
    enable_deprecation_warnings()

    _warn_deprecated(
        "`train()` is deprecated and will be removed in v0.14. Use `run()` instead."
    )

    run(recipe, config, output_dir)


def evaluate(recipe: Recipe, config: object, output_dir: Path) -> None:
    enable_deprecation_warnings()

    _warn_deprecated(
        "`evaluate()` is deprecated and will be removed in v0.14. Use `run()` instead."
    )

    run(recipe, config, output_dir)


def generate(recipe: Recipe, config: object, output_dir: Path) -> None:
    enable_deprecation_warnings()

    _warn_deprecated(
        "`generate()` is deprecated and will be removed in v0.14. Use `run()` instead."
    )

    run(recipe, config, output_dir)


#
# DEPRECATED - END
#


def run(
    recipe: Recipe,
    config: object,
    output_dir: Path,
    *,
    no_rich: bool = False,
    no_progress: bool | None = None,
) -> None:
    from fairseq2.recipe.composition import (
        _register_inference_recipe,
        _register_train_recipe,
    )

    enable_deprecation_warnings()

    try:
        configure_logging(no_rich=no_rich)
    except EnvironmentVariableError as ex:
        raise ConfigError(f"Failed to initalize logging. {ex}") from None

    is_train_recipe = _is_train_config(recipe.config_kls)

    container = DependencyContainer()

    with _swap_default_resolver(container):
        with torch.inference_mode(mode=not is_train_recipe):
            _register_library(container, no_progress=True if no_rich else no_progress)

            if is_train_recipe:
                _register_train_recipe(container, recipe)
            else:
                _register_inference_recipe(container, recipe)

            _register_run(container, recipe, config, output_dir)

            _run_recipe(container)


@contextmanager
def _swap_default_resolver(resolver: DependencyResolver) -> Iterator[None]:
    original_resolver = fairseq2.runtime.dependency._resolver

    fairseq2.runtime.dependency._resolver = resolver

    try:
        yield
    finally:
        fairseq2.runtime.dependency._resolver = original_resolver


def _register_run(
    container: DependencyContainer, recipe: Recipe, config: object, output_dir: Path
) -> None:
    config_kls = recipe.config_kls

    if not isinstance(config, config_kls):
        raise TypeError(
            f"`config` must be of type `{config_kls}`, but is of type `{type(config)}` instead."
        )

    # Recipe Configuration
    def get_config(resolver: DependencyResolver) -> _RecipeConfigHolder:
        validator = resolver.resolve(ObjectValidator)

        validator.validate(config)

        return _RecipeConfigHolder(config)

    container.register(_RecipeConfigHolder, get_config)

    # Recipe Output Directory
    def get_output_dir(resolver: DependencyResolver) -> Path:
        dir_creator = resolver.resolve(_OutputDirectoryCreator)

        return dir_creator.create(output_dir)

    container.register(Path, get_output_dir)


def _run_recipe(resolver: DependencyResolver) -> None:
    # Prepare cluster environment.
    cluster_preparer = resolver.resolve(_ClusterPreparer)

    cluster_preparer.prepare()

    # Configure distributed logging.
    log_configurer = resolver.resolve(_DistributedLogConfigurer)

    log_configurer.configure()

    # Save config to file.
    config_dumper = resolver.resolve(_RecipeConfigDumper)

    config_dumper.dump()

    # Configure PyTorch.
    torch_configurer = resolver.resolve(_TorchConfigurer)

    torch_configurer.configure()

    # Log environment info.
    log_helper = resolver.resolve(_LogHelper)

    log_helper.log_system_info()

    log_helper.log_software_info()

    log_helper.log_environment_variables()

    # Load assets.
    try:
        activate_dependency(resolver, AssetStore)
    except EnvironmentVariableError as ex:
        raise CommandError(f"Failed to load asset store. {ex}") from None
    except AssetMetadataSourceNotFoundError as ex:
        raise CommandError(
            f"Failed to load asset store. {ex.source} asset metadata source is not found."
        ) from None
    except CorruptAssetMetadataError as ex:
        raise CommandError(
            f"Failed to load asset store. {ex.source} asset metadata source is corrupt."
        ) from ex
    except AssetMetadataLoadError as ex:
        raise OperationalError("Failed to load asset store.") from ex

    # TODO: activate AssetDownloadManager!

    # Run recipe task.
    task = resolver.resolve(Task)

    task_runner = resolver.resolve(_TaskRunner)

    task_runner.run(task)


@final
class _RecipeConfigDumper:
    def __init__(
        self,
        config_holder: _RecipeConfigHolder,
        output_dir: Path,
        world_info: WorldInfo,
        log_helper: _LogHelper,
        value_converter: ValueConverter,
        yaml_dumper: YamlDumper,
    ) -> None:
        self._config_holder = config_holder
        self._output_dir = output_dir
        self._world_info = world_info
        self._log_helper = log_helper
        self._value_converter = value_converter
        self._yaml_dumper = yaml_dumper

    def dump(self) -> None:
        self._log_helper.log_config("Config", self._config_holder.config)

        if self._world_info.rank != 0:
            return

        try:
            unstructured_config = self._value_converter.unstructure(
                self._config_holder.config
            )
        except StructureError as ex:
            raise InternalError("Recipe configuration cannot be unstructured.") from ex

        file = self._output_dir.joinpath("config.yaml")

        try:
            self._yaml_dumper.dump(unstructured_config, file)
        except YamlError as ex:
            raise InternalError(
                "Recipe configuration cannot be serialized to YAML."
            ) from ex
        except OSError as ex:
            raise OperationalError(
                f"Failed to dump recipe configuration to {file}."
            ) from ex
