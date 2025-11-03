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

from typing_extensions import override

from fairseq2.assets import AssetStore
from fairseq2.checkpoint import CheckpointManager, ModelMetadataDumper
from fairseq2.error import InternalError, raise_operational_system_error
from fairseq2.gang import GangError, Gangs, raise_operational_gang_error
from fairseq2.logging import log
from fairseq2.models import (
    ModelArchitectureNotKnownError,
    ModelFamily,
    ModelFamilyNotKnownError,
    ModelNotKnownError,
    get_model_family,
)
from fairseq2.recipe.config import ModelSection, TrainerSection
from fairseq2.recipe.error import (
    LayerwiseACNotSupportedError,
    ModelCheckpointNotFoundError,
)
from fairseq2.recipe.internal.asset_config import _AssetConfigOverrider
from fairseq2.recipe.internal.compile import _compile_model
from fairseq2.recipe.internal.data_parallel import _DataParallelModelWrapper
from fairseq2.recipe.internal.log import _log_model, _LogHelper
from fairseq2.recipe.internal.model import _ModelHolder
from fairseq2.runtime.lookup import Lookup


@final
class _TrainModelProvider:
    def __init__(
        self,
        bootstrapper: _TrainModelBootstrapper,
        preparer: _TrainModelPreparer,
        gangs: Gangs,
    ) -> None:
        self._bootstrapper = bootstrapper
        self._preparer = preparer
        self._gangs = gangs

    def get(self) -> _ModelHolder:
        model_holder = self._bootstrapper.bootstrap()

        self._preparer.prepare(model_holder)

        _log_model(model_holder.dp_model, self._gangs)

        return model_holder


class _TrainModelBootstrapper(ABC):
    @abstractmethod
    def bootstrap(self) -> _ModelHolder: ...


@final
class _StandardTrainModelBootstrapper(_TrainModelBootstrapper):
    def __init__(
        self,
        section: ModelSection,
        families: Lookup[ModelFamily],
        asset_store: AssetStore,
        asset_config_overrider: _AssetConfigOverrider,
        checkpoint_manager: CheckpointManager,
        metadata_saver: _TrainModelMetadataSaver,
        gangs: Gangs,
        log_helper: _LogHelper,
    ) -> None:
        self._section = section
        self._families = families
        self._asset_store = asset_store
        self._asset_config_overrider = asset_config_overrider
        self._checkpoint_manager = checkpoint_manager
        self._metadata_saver = metadata_saver
        self._gangs = gangs
        self._log_helper = log_helper

    @override
    def bootstrap(self) -> _ModelHolder:
        section = self._section

        if section.path is not None:
            if section.name is not None:
                log.warning("Both `model.name` and `model.path` are specified. `model.path` takes precedence.")  # fmt: skip

            return self._load_custom_model()

        if section.name is not None:
            if section.family is not None:
                log.warning("`model.family` will be ignored since `model.name` is specifed.")  # fmt: skip

            return self._load_model()

        if section.family is not None:
            return self._create_new_model()

        raise InternalError("`section.name` and `section.family` are both `None`.")

    def _load_model(self) -> _ModelHolder:
        name = self._section.name
        if name is None:
            raise InternalError("`section.name` is `None`.")

        card = self._asset_store.maybe_retrieve_card(name)
        if card is None:
            raise ModelNotKnownError(name)

        family = get_model_family(card, self._families)

        config = family.get_model_config(card)

        if self._section.config_overrides is not None:
            config = self._asset_config_overrider.apply_overrides(
                "model", config, self._section.config_overrides
            )

        gangs = self._gangs

        # Load the model.
        dtype = self._section.dtype

        has_checkpoint = self._checkpoint_manager.has_checkpoint(
            exclude_model_only=True
        )

        if has_checkpoint:
            log.info("Initializing checkpoint model.")

            if config is not None:
                self._log_helper.log_config("Model Config", config)

            model = family.create_new_model(
                config, gangs, dtype, meta=family.supports_meta
            )

            if not family.supports_meta:
                try:
                    gangs.root.barrier()
                except GangError as ex:
                    raise_operational_gang_error(ex)

            log.info("Model initialized.")
        else:
            log.info("Loading {} model on data parallel rank 0.", name)

            if config is not None:
                self._log_helper.log_config("Model Config", config)

            if gangs.dp.rank == 0:
                model = family.load_model(
                    card, gangs, dtype, config, self._section.mmap, progress=True
                )
            else:
                model = family.create_new_model(
                    config, gangs, dtype, meta=family.supports_meta
                )

            try:
                gangs.root.barrier()
            except GangError as ex:
                raise_operational_gang_error(ex)

            log.info("Model loaded.")

        model.requires_grad_(True)

        model.train()

        self._metadata_saver.save(family.name, config)

        return _ModelHolder(model, family, config)

    def _load_custom_model(self) -> _ModelHolder:
        path = self._section.path
        if path is None:
            raise InternalError("`section.path` is `None`.")

        family_name = self._section.family
        if family_name is None:
            raise InternalError("`section.family` is `None`.")

        family = self._families.maybe_get(family_name)
        if family is None:
            raise ModelFamilyNotKnownError(family_name)

        arch = self._section.arch
        if arch is None:
            try:
                config = family.config_kls()
            except TypeError as ex:
                raise InternalError(
                    f"Default configuration of the {family} model family cannot be constructed."
                ) from ex
        else:
            config = family.maybe_get_arch_config(arch)
            if config is None:
                raise ModelArchitectureNotKnownError(arch, family_name) from None

        if self._section.config_overrides is not None:
            config = self._asset_config_overrider.apply_overrides(
                "model", config, self._section.config_overrides
            )

        gangs = self._gangs

        # Load the model.
        dtype = self._section.dtype

        has_checkpoint = self._checkpoint_manager.has_checkpoint(
            exclude_model_only=True
        )

        if has_checkpoint:
            log.info("Initializing checkpoint model.")

            if config is not None:
                self._log_helper.log_config("Model Config", config)

            model = family.create_new_model(
                config, gangs, dtype, meta=family.supports_meta
            )

            if not family.supports_meta:
                try:
                    gangs.root.barrier()
                except GangError as ex:
                    raise_operational_gang_error(ex)

            log.info("Model initialized.")
        else:
            log.info("Loading model on data parallel rank 0.")

            if config is not None:
                self._log_helper.log_config("Model Config", config)

            if gangs.dp.rank == 0:
                try:
                    model = family.load_custom_model(
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
                model = family.create_new_model(
                    config, gangs, dtype, meta=family.supports_meta
                )

            try:
                gangs.root.barrier()
            except GangError as ex:
                raise_operational_gang_error(ex)

            log.info("Model loaded on data parallel rank 0.")

        model.requires_grad_(True)

        model.train()

        self._metadata_saver.save(family.name, config)

        return _ModelHolder(model, family, config)

    def _create_new_model(self) -> _ModelHolder:
        family_name = self._section.family
        if family_name is None:
            raise InternalError("`section.family` is `None`.")

        family = self._families.maybe_get(family_name)
        if family is None:
            raise ModelFamilyNotKnownError(family_name)

        arch = self._section.arch
        if arch is None:
            try:
                config = family.config_kls()
            except TypeError as ex:
                raise InternalError(
                    f"Default configuration of the {family} model family cannot be constructed."
                ) from ex
        else:
            config = family.maybe_get_arch_config(arch)
            if config is None:
                raise ModelArchitectureNotKnownError(arch, family_name) from None

        if self._section.config_overrides is not None:
            config = self._asset_config_overrider.apply_overrides(
                "model", config, self._section.config_overrides
            )

        gangs = self._gangs

        # Create the model.
        dtype = self._section.dtype

        has_checkpoint = self._checkpoint_manager.has_checkpoint(
            exclude_model_only=True
        )

        if has_checkpoint:
            log.info("Initializing checkpoint model.")

            if config is not None:
                self._log_helper.log_config("Model Config", config)

            model = family.create_new_model(
                config, gangs, dtype, meta=family.supports_meta
            )

            if not family.supports_meta:
                try:
                    gangs.root.barrier()
                except GangError as ex:
                    raise_operational_gang_error(ex)

            log.info("Model initialized.")
        else:
            if not family.supports_meta:
                log.info("Initializing model.")
            else:
                log.info("Initializing model on data parallel rank 0.")

            if config is not None:
                self._log_helper.log_config("Model Config", config)

            if gangs.dp.rank == 0:
                meta = False
            else:
                meta = family.supports_meta

            model = family.create_new_model(config, gangs, dtype, meta)

            try:
                gangs.root.barrier()
            except GangError as ex:
                raise_operational_gang_error(ex)

            log.info("Model initialized.")

        model.requires_grad_(True)

        model.train()

        self._metadata_saver.save(family.name, config)

        return _ModelHolder(model, family, config, newly_initialized=not has_checkpoint)


class _TrainModelMetadataSaver(ABC):
    @abstractmethod
    def save(self, family_name: str, config: object) -> None: ...


@final
class _StandardTrainModelMetadataSaver(_TrainModelMetadataSaver):
    def __init__(
        self, metadata_dumper: ModelMetadataDumper, output_dir: Path, gangs: Gangs
    ) -> None:
        self._metadata_dumper = metadata_dumper
        self._output_dir = output_dir
        self._gangs = gangs

    @override
    def save(self, family_name: str, config: object) -> None:
        checkpoint_dir = self._output_dir.joinpath("checkpoints")

        if self._gangs.root.rank == 0:
            try:
                self._metadata_dumper.dump(checkpoint_dir, family_name, config)
            except OSError as ex:
                raise_operational_system_error(ex)

        try:
            self._gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)


class _TrainModelPreparer(ABC):
    @abstractmethod
    def prepare(self, model_holder: _ModelHolder) -> None: ...


@final
class _DelegatingTrainModelPreparer(_TrainModelPreparer):
    def __init__(self, preparers: Iterable[_TrainModelPreparer]) -> None:
        self._preparers = preparers

    @override
    def prepare(self, model_holder: _ModelHolder) -> None:
        for preparer in self._preparers:
            preparer.prepare(model_holder)


@final
class _LastTrainModelPreparer(_TrainModelPreparer):
    def __init__(
        self,
        section: ModelSection,
        trainer_section: TrainerSection,
        data_parallel_wrapper: _DataParallelModelWrapper,
    ) -> None:
        self._section = section
        self._trainer_section = trainer_section
        self._data_parallel_wrapper = data_parallel_wrapper

    @override
    def prepare(self, model_holder: _ModelHolder) -> None:
        ac_config = self._trainer_section.activation_checkpointing

        # Apply AC before torch.compile() so that min-cut partitioner can see
        # the AC information and avoid recomputing twice.
        if ac_config.mode == "layerwise":
            if not model_holder.family.supports_layerwise_ac:
                raise LayerwiseACNotSupportedError()

            model_holder.family.apply_layerwise_ac(
                model_holder.model, ac_config.every_nth_layer
            )

        if self._section.compile:
            _compile_model(model_holder, "model", self._section.compile_options)

        model_holder.dp_model = self._data_parallel_wrapper.wrap(model_holder)
