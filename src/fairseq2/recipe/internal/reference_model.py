# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import final

from typing_extensions import override

from fairseq2.assets import AssetStore
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
from fairseq2.nn.utils.module import broadcast_module, remove_parametrizations
from fairseq2.recipe.config import ReferenceModelSection
from fairseq2.recipe.error import ModelCheckpointNotFoundError
from fairseq2.recipe.internal.asset_config import _AssetConfigOverrider
from fairseq2.recipe.internal.compile import _compile_model
from fairseq2.recipe.internal.log import _log_model, _LogHelper
from fairseq2.recipe.internal.model import _ModelHolder
from fairseq2.runtime.lookup import Lookup


@final
class _ReferenceModelLoader:
    def __init__(
        self,
        bootstrapper: _ReferenceModelBootstrapper,
        preparer: _ReferenceModelPreparer,
        gangs: Gangs,
    ) -> None:
        self._bootstrapper = bootstrapper
        self._preparer = preparer
        self._gangs = gangs

    def load(self, section_name: str, section: ReferenceModelSection) -> _ModelHolder:
        model_holder = self._bootstrapper.bootstrap(section_name, section)

        self._preparer.prepare(model_holder, section_name, section)

        _log_model(model_holder.model, self._gangs)

        return model_holder


class _ReferenceModelBootstrapper(ABC):
    @abstractmethod
    def bootstrap(
        self, section_name: str, section: ReferenceModelSection
    ) -> _ModelHolder: ...


@final
class _StandardReferenceModelBootstrapper(_ReferenceModelBootstrapper):
    def __init__(
        self,
        families: Lookup[ModelFamily],
        asset_store: AssetStore,
        asset_config_overrider: _AssetConfigOverrider,
        gangs: Gangs,
        log_helper: _LogHelper,
    ) -> None:
        self._families = families
        self._asset_store = asset_store
        self._asset_config_overrider = asset_config_overrider
        self._gangs = gangs
        self._log_helper = log_helper

    @override
    def bootstrap(
        self, section_name: str, section: ReferenceModelSection
    ) -> _ModelHolder:
        if section.path is not None:
            if section.name is not None:
                log.warning("Both `{0}.name` and `{0}.path` are specified. `{0}.path` takes precedence.", section_name)  # fmt: skip

            return self._load_custom_model(section_name, section)

        if section.name is not None:
            if section.family is not None:
                log.warning("`{0}.family` will be ignored since `{0}.name` is specified.", section_name)  # fmt: skip

            return self._load_model(section_name, section)

        raise InternalError("`section.name` and `section.family` are both `None`.")

    def _load_model(
        self, section_name: str, section: ReferenceModelSection
    ) -> _ModelHolder:
        name = section.name
        if name is None:
            raise InternalError("`section.name` is `None`.")

        card = self._asset_store.maybe_retrieve_card(name)
        if card is None:
            raise ModelNotKnownError(name)

        family = get_model_family(card, self._families)

        config = family.get_model_config(card)

        config = self._asset_config_overrider.apply_overrides(
            section_name, config, section.config_overrides
        )

        gangs = self._gangs

        # Load the model.
        if section_name == "model":
            log.info("Loading {} model on data parallel rank 0.", name)
        else:
            log.info("Loading {} model specified in `{}` section on data parallel rank 0.", name, section_name)  # fmt: skip

        self._log_helper.log_config("Model Config", config)

        model = family.load_model(
            card, gangs, section.dtype, config, load_rank0_only=True, mmap=section.mmap
        )

        log.info("Model loaded on data parallel rank 0.")

        model.requires_grad_(False)

        model.eval()

        return _ModelHolder(model, family, config)

    def _load_custom_model(
        self, section_name: str, section: ReferenceModelSection
    ) -> _ModelHolder:
        path = section.path
        if path is None:
            raise InternalError("`section.path` is `None`.")

        family_name = section.family
        if family_name is None:
            raise InternalError("`section.family` is `None`.")

        family = self._families.maybe_get(family_name)
        if family is None:
            raise ModelFamilyNotKnownError(family_name)

        arch = section.arch
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

        config = self._asset_config_overrider.apply_overrides(
            section_name, config, section.config_overrides
        )

        gangs = self._gangs

        # Load the model.
        if section_name == "model":
            log.info("Loading model on data parallel rank 0.")
        else:
            log.info("Loading model specified in `{}` section on data parallel rank 0.", section_name)  # fmt: skip

        self._log_helper.log_config("Model Config", config)

        try:
            model = family.load_custom_model(
                path,
                config,
                gangs,
                section.dtype,
                load_rank0_only=True,
                mmap=section.mmap,
                restrict=None,
            )
        except FileNotFoundError as ex:
            raise ModelCheckpointNotFoundError(path) from ex
        except OSError as ex:
            raise_operational_system_error(ex)

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

        log.info("Model loaded on data parallel rank 0.")

        model.requires_grad_(False)

        model.eval()

        return _ModelHolder(model, family, config)


class _ReferenceModelPreparer(ABC):
    @abstractmethod
    def prepare(
        self,
        model: _ModelHolder,
        section_name: str,
        section: ReferenceModelSection,
    ) -> None: ...


@final
class _DelegatingReferenceModelPreparer(_ReferenceModelPreparer):
    def __init__(self, preparers: Iterable[_ReferenceModelPreparer]) -> None:
        self._preparers = preparers

    @override
    def prepare(
        self,
        model_holder: _ModelHolder,
        section_name: str,
        section: ReferenceModelSection,
    ) -> None:
        for preparer in self._preparers:
            preparer.prepare(model_holder, section_name, section)


@final
class _LastReferenceModelPreparer(_ReferenceModelPreparer):
    def __init__(self, gangs: Gangs) -> None:
        self._gangs = gangs

    @override
    def prepare(
        self,
        model_holder: _ModelHolder,
        section_name: str,
        section: ReferenceModelSection,
    ) -> None:
        if self._gangs.dp.size > 1:
            log.info("Broadcasting model to all processes.")

            try:
                broadcast_module(model_holder.model, self._gangs.dp)
            except GangError as ex:
                raise_operational_gang_error(ex)

            log.info("Model broadcasted.")

        remove_parametrizations(model_holder.model)

        if section.compile:
            _compile_model(model_holder, section_name, section.compile_options)
