# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import final

import torch
from typing_extensions import override

from fairseq2.assets import AssetCardError, AssetStore
from fairseq2.checkpoint import CheckpointManager, ModelMetadataDumper
from fairseq2.error import InternalError, raise_operational_system_error
from fairseq2.gang import GangError, Gangs, raise_operational_gang_error
from fairseq2.logging import log
from fairseq2.model import Model, StandardModel
from fairseq2.models import (
    ModelFamilyHandler,
    ModelFamilyNotKnownError,
    ModelNotKnownError,
)
from fairseq2.recipe.asset_config import AssetConfigOverrider
from fairseq2.recipe.base import RecipeContext, TrainRecipe
from fairseq2.recipe.compile import _compile_model
from fairseq2.recipe.config import ModelSection, TrainerSection
from fairseq2.recipe.data_parallel import DPModelWrapper
from fairseq2.runtime.config_registry import ConfigNotFoundError
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.runtime.provider import Provider
from fairseq2.utils.log import log_config, log_model


@final
class ModelFactory:
    def __init__(
        self,
        bootstrapper: ModelBootstrapper,
        metadata_saver: ModelMetadataSaver,
        preparer: ModelPreparer,
        gangs: Gangs,
    ) -> None:
        self._bootstrapper = bootstrapper
        self._metadata_saver = metadata_saver
        self._preparer = preparer
        self._gangs = gangs

    def create(self) -> Model:
        model = self._bootstrapper.bootstrap()

        self._metadata_saver.save(model)

        model = self._preparer.prepare(model)

        log_model(model.module, self._gangs)

        return model


class ModelBootstrapper(ABC):
    @abstractmethod
    def bootstrap(self) -> Model: ...


@final
class StandardModelBootstrapper(ModelBootstrapper):
    def __init__(
        self,
        section: ModelSection,
        trainer_section: TrainerSection,
        handlers: Provider[ModelFamilyHandler],
        asset_store: AssetStore,
        asset_config_overrider: AssetConfigOverrider,
        checkpoint_manager: CheckpointManager,
        gangs: Gangs,
    ) -> None:
        self._section = section
        self._trainer_section = trainer_section
        self._handlers = handlers
        self._asset_store = asset_store
        self._asset_config_overrider = asset_config_overrider
        self._checkpoint_manager = checkpoint_manager
        self._gangs = gangs

    @override
    def bootstrap(self) -> Model:
        if self._section.path is not None:
            return self._load_model_from_path()

        if self._section.name is not None:
            return self._load_model_from_card()

        if self._section.family is not None:
            return self._create_new_model()

        raise InternalError("`section.name` and `section.family` are both `None`.")

    def _load_model_from_card(self) -> Model:
        name = self._section.name
        if name is None:
            raise InternalError("`section.name` is `None`.")

        card = self._asset_store.maybe_retrieve_card(name)
        if card is None:
            raise ModelNotKnownError(name)

        family = card.field("model_family").as_(str)

        handler = self._handlers.maybe_get(family)
        if handler is None:
            msg = f"family field of the {name} asset card is expected to be a supported model family, but is {family} instead."

            raise AssetCardError(name, msg)

        config = handler.get_model_config(card)

        if self._section.config_overrides is not None:
            config = self._asset_config_overrider.apply_overrides(
                "model", config, self._section.config_overrides
            )

        self._log_model_config(name, config)

        gangs = self._gangs

        if gangs.root.size != gangs.dp.size:
            if not handler.supports_model_parallelism:
                raise ModelParallelismNotSupportedError(name)

        # Load the model.
        if self._trainer_section.mixed_precision == "off":
            dtype = self._trainer_section.dtype
        else:
            dtype = torch.float32

        has_checkpoint = self._checkpoint_manager.has_checkpoint(
            exclude_model_only=True
        )

        if has_checkpoint:
            if not handler.supports_meta:
                log.info("Initializing {} model.", name)

            module = handler.create_new_model(
                config, gangs, dtype, meta=handler.supports_meta
            )

            if not handler.supports_meta:
                try:
                    gangs.root.barrier()
                except GangError as ex:
                    raise_operational_gang_error(ex)

                log.info("Model initialized.")
        else:
            log.info("Loading {} model on data parallel rank 0.", name)

            if gangs.dp.rank == 0:
                module = handler.load_model(
                    card, gangs, dtype, config, self._section.mmap, progress=True
                )
            else:
                module = handler.create_new_model(
                    config, gangs, dtype, meta=handler.supports_meta
                )

            try:
                gangs.root.barrier()
            except GangError as ex:
                raise_operational_gang_error(ex)

            log.info("Model loaded.")

        module.train()

        return StandardModel(name, module, config, handler)

    def _load_model_from_path(self) -> Model:
        path = self._section.path
        if path is None:
            raise InternalError("`section.path` is `None`.")

        family = self._section.family
        if family is None:
            raise InternalError("`section.family` is `None`.")

        name = path.name

        handler = self._handlers.maybe_get(family)
        if handler is None:
            raise ModelFamilyNotKnownError(family)

        arch = self._section.arch
        if arch is None:
            try:
                config = handler.config_kls()
            except TypeError as ex:
                raise InternalError(
                    f"Default configuration of the {family} model family cannot be constructed."
                ) from ex
        else:
            try:
                config = handler.get_arch_config(arch)
            except ConfigNotFoundError:
                raise ModelArchitectureNotKnownError(arch) from None

        if self._section.config_overrides is not None:
            config = self._asset_config_overrider.apply_overrides(
                "model", config, self._section.config_overrides
            )

        self._log_model_config(name, config)

        gangs = self._gangs

        if gangs.root.size != gangs.dp.size:
            if not handler.supports_model_parallelism:
                raise ModelParallelismNotSupportedError(name)

        # Load the model.
        if self._trainer_section.mixed_precision == "off":
            dtype = self._trainer_section.dtype
        else:
            dtype = torch.float32

        has_checkpoint = self._checkpoint_manager.has_checkpoint(
            exclude_model_only=True
        )

        if has_checkpoint:
            if not handler.supports_meta:
                log.info("Initializing {} model.", name)

            module = handler.create_new_model(
                config, gangs, dtype, meta=handler.supports_meta
            )

            if not handler.supports_meta:
                try:
                    gangs.root.barrier()
                except GangError as ex:
                    raise_operational_gang_error(ex)

                log.info("Model initialized.")
        else:
            log.info("Loading {} model on data parallel rank 0.", name)

            if gangs.dp.rank == 0:
                try:
                    module = handler.load_custom_model(
                        path,
                        config,
                        gangs,
                        dtype,
                        self._section.mmap,
                        restrict=None,
                        progress=True,
                    )
                except FileNotFoundError as ex:
                    raise ModelCheckpointNotFoundError(path) from ex
                except OSError as ex:
                    raise_operational_system_error(ex)
            else:
                module = handler.create_new_model(
                    config, gangs, dtype, meta=handler.supports_meta
                )

            try:
                gangs.root.barrier()
            except GangError as ex:
                raise_operational_gang_error(ex)

            log.info("Model loaded.")

        module.train()

        return StandardModel(name, module, config, handler)

    def _create_new_model(self) -> Model:
        family = self._section.family
        if family is None:
            raise InternalError("`section.family` is `None`.")

        name = "train"

        handler = self._handlers.maybe_get(family)
        if handler is None:
            raise ModelFamilyNotKnownError(family)

        arch = self._section.arch
        if arch is None:
            try:
                config = handler.config_kls()
            except TypeError as ex:
                raise InternalError(
                    f"Default configuration of the {family} model family cannot be constructed."
                ) from ex
        else:
            try:
                config = handler.get_arch_config(arch)
            except ConfigNotFoundError:
                raise ModelArchitectureNotKnownError(arch) from None

        if self._section.config_overrides is not None:
            config = self._asset_config_overrider.apply_overrides(
                "model", config, self._section.config_overrides
            )

        self._log_model_config(name, config)

        gangs = self._gangs

        if gangs.root.size != gangs.dp.size:
            if not handler.supports_model_parallelism:
                raise ModelParallelismNotSupportedError(name)

        # Create the model.
        if self._trainer_section.mixed_precision == "off":
            dtype = self._trainer_section.dtype
        else:
            dtype = torch.float32

        has_checkpoint = self._checkpoint_manager.has_checkpoint(
            exclude_model_only=True
        )

        if has_checkpoint:
            if not handler.supports_meta:
                log.info("Initializing model.")

            model = handler.create_new_model(
                config, gangs, dtype, meta=handler.supports_meta
            )

            if not handler.supports_meta:
                try:
                    gangs.root.barrier()
                except GangError as ex:
                    raise_operational_gang_error(ex)

                log.info("Model initialized.")
        else:
            if not handler.supports_meta:
                log.info("Initializing model.")
            else:
                log.info("Initializing model on data parallel rank 0.")

            if gangs.dp.rank == 0:
                meta = False
            else:
                meta = handler.supports_meta

            model = handler.create_new_model(config, gangs, dtype, meta)

            try:
                gangs.root.barrier()
            except GangError as ex:
                raise_operational_gang_error(ex)

            log.info("Model initialized.")

        model.train()

        return StandardModel(
            name, model, config, handler, newly_initialized=not has_checkpoint
        )

    def _log_model_config(self, name: str, config: object) -> None:
        if not log.is_enabled_for_info():
            return

        title = "Model Config" if name == "train" else f"{name} Model Config"

        log_config(title, config)


class ModelMetadataSaver(ABC):
    @abstractmethod
    def save(self, model: Model) -> None: ...


@final
class StandardModelMetadataSaver(ModelMetadataSaver):
    def __init__(
        self, metadata_dumper: ModelMetadataDumper, output_dir: Path, gangs: Gangs
    ) -> None:
        self._metadata_dumper = metadata_dumper
        self._output_dir = output_dir
        self._gangs = gangs

    @override
    def save(self, model: Model) -> None:
        checkpoint_dir = self._output_dir.joinpath("checkpoints")

        if self._gangs.root.rank == 0:
            try:
                self._metadata_dumper.dump(checkpoint_dir, model)
            except OSError as ex:
                raise_operational_system_error(ex)

        try:
            self._gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)


class ModelPreparer(ABC):
    @abstractmethod
    def prepare(self, model: Model) -> Model: ...


@final
class DelegatingModelPreparer(ModelPreparer):
    def __init__(self, preparers: Iterable[ModelPreparer]) -> None:
        self._preparers = list(preparers)

    @override
    def prepare(self, model: Model) -> Model:
        for preparer in self._preparers:
            model = preparer.prepare(model)

        return model


@final
class StandardModelPreparer(ModelPreparer):
    def __init__(
        self,
        section: ModelSection,
        trainer_section: TrainerSection,
        data_parallel_wrapper: DPModelWrapper,
    ) -> None:
        self._section = section
        self._trainer_section = trainer_section
        self._data_parallel_wrapper = data_parallel_wrapper

    @override
    def prepare(self, model: Model) -> Model:
        ac = self._trainer_section.activation_checkpointing

        # Apply AC before torch.compile() so that min-cut partitioner can see the AC
        # information and avoid recomputing twice.
        if ac.mode == "layerwise":
            if not model.handler.supports_activation_checkpointing:
                raise ActivationCheckpointingNotSupportedError(model.name)

            model.handler.apply_activation_checkpointing(
                model.module, every_nth_layer=ac.every_nth_layer
            )

        if self._section.compile:
            _compile_model(model, self._section.compile_options)

        model = self._data_parallel_wrapper.wrap(model)

        return model


@final
class RecipeModelPreparer(ModelPreparer):
    def __init__(self, recipe: TrainRecipe, resolver: DependencyResolver) -> None:
        self._recipe = recipe
        self._resolver = resolver

    @override
    def prepare(self, model: Model) -> Model:
        context = RecipeContext(self._resolver)

        return self._recipe.prepare_model(context, model)


class ModelArchitectureNotKnownError(Exception):
    def __init__(self, arch: str) -> None:
        super().__init__(f"{arch} is not a known model architecture.")

        self.arch = arch


class ModelParallelismNotSupportedError(Exception):
    def __init__(self, model_name: str) -> None:
        super().__init__(f"{model_name} model does not support model parallelism.")

        self.model_name = model_name


class ModelCheckpointNotFoundError(Exception):
    def __init__(self, path: Path) -> None:
        super().__init__(f"{path} does not point to a model checkpoint.")

        self.path = path


class ActivationCheckpointingNotSupportedError(Exception):
    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"{model_name} model does not support activation checkpointing."
        )

        self.model_name = model_name
