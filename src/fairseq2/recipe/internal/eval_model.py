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
from fairseq2.recipe.error import ErrorContext, ModelCheckpointNotFoundError
from fairseq2.recipe.internal.asset_config import _AssetConfigOverrider
from fairseq2.recipe.internal.compile import _compile_model
from fairseq2.recipe.internal.log import _LogHelper
from fairseq2.recipe.internal.model import _log_model
from fairseq2.recipe.model import RecipeModel, _StandardRecipeModel
from fairseq2.runtime.lookup import Lookup


@final
class _EvalModelLoader:
    def __init__(
        self,
        bootstrapper: _EvalModelBootstrapper,
        preparer: _EvalModelPreparer,
        gangs: Gangs,
    ) -> None:
        self._bootstrapper = bootstrapper
        self._preparer = preparer
        self._gangs = gangs

    def load(self, section_name: str, section: ReferenceModelSection) -> RecipeModel:
        try:
            model = self._bootstrapper.bootstrap(section_name, section)

            model = self._preparer.prepare(model, section_name, section)
        except Exception as ex:
            ErrorContext.set_config_section_name(ex, section_name)

            raise

        _log_model(model, self._gangs)

        return model


class _EvalModelBootstrapper(ABC):
    @abstractmethod
    def bootstrap(
        self, section_name: str, section: ReferenceModelSection
    ) -> RecipeModel: ...


@final
class _StandardEvalModelBootstrapper(_EvalModelBootstrapper):
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
    ) -> RecipeModel:
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
    ) -> RecipeModel:
        name = section.name
        if name is None:
            raise InternalError("`section.name` is `None`.")

        card = self._asset_store.maybe_retrieve_card(name)
        if card is None:
            raise ModelNotKnownError(name)

        family = get_model_family(card, self._families)

        config = family.get_model_config(card)

        if section.config_overrides is not None:
            config = self._asset_config_overrider.apply_overrides(
                section_name, config, section.config_overrides
            )

        gangs = self._gangs

        # Load the model.
        if section_name == "model":
            log.info("Loading {} model on data parallel rank 0.", name)
        else:
            log.info("Loading {} model specified in `{}` section on data parallel rank 0.", name, section_name)  # fmt: skip

        if config is not None:
            self._log_helper.log_config("Model Config", config)

        if gangs.dp.rank == 0:
            module = family.load_model(
                card, gangs, section.dtype, config, section.mmap, progress=True
            )
        else:
            module = family.create_new_model(
                config, gangs, section.dtype, meta=family.supports_meta
            )

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

        log.info("Model loaded on data parallel rank 0.")

        module.requires_grad_(False)

        module.eval()

        return _StandardRecipeModel(module, config, family)

    def _load_custom_model(
        self, section_name: str, section: ReferenceModelSection
    ) -> RecipeModel:
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

        if section.config_overrides is not None:
            config = self._asset_config_overrider.apply_overrides(
                section_name, config, section.config_overrides
            )

        gangs = self._gangs

        # Load the model.
        if section_name == "model":
            log.info("Loading model on data parallel rank 0.")
        else:
            log.info("Loading model specified in `{}` section on data parallel rank 0.", section_name)  # fmt: skip

        if config is not None:
            self._log_helper.log_config("Model Config", config)

        if gangs.dp.rank == 0:
            try:
                module = family.load_custom_model(
                    path,
                    config,
                    gangs,
                    section.dtype,
                    section.mmap,
                    restrict=None,
                    progress=True,
                )
            except FileNotFoundError as ex:
                raise ModelCheckpointNotFoundError(path) from ex
            except OSError as ex:
                raise_operational_system_error(ex)
        else:
            module = family.create_new_model(
                config, gangs, section.dtype, meta=family.supports_meta
            )

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

        log.info("Model loaded on data parallel rank 0.")

        module.requires_grad_(False)

        module.eval()

        return _StandardRecipeModel(module, config, family)


class _EvalModelPreparer(ABC):
    @abstractmethod
    def prepare(
        self, model: RecipeModel, section_name: str, section: ReferenceModelSection
    ) -> RecipeModel: ...


@final
class _DelegatingEvalModelPreparer(_EvalModelPreparer):
    def __init__(self, preparers: Iterable[_EvalModelPreparer]) -> None:
        self._preparers = preparers

    @override
    def prepare(
        self, model: RecipeModel, section_name: str, section: ReferenceModelSection
    ) -> RecipeModel:
        for preparer in self._preparers:
            model = preparer.prepare(model, section_name, section)

        return model


@final
class _StandardEvalModelPreparer(_EvalModelPreparer):
    def __init__(self, gangs: Gangs) -> None:
        self._gangs = gangs

    @override
    def prepare(
        self, model: RecipeModel, section_name: str, section: ReferenceModelSection
    ) -> RecipeModel:
        if self._gangs.dp.size > 1:
            log.info("Broadcasting model to all processes.")

            try:
                broadcast_module(model.module, self._gangs.dp)
            except GangError as ex:
                raise_operational_gang_error(ex)

            log.info("Model broadcasted.")

        remove_parametrizations(model.module)

        if section.compile:
            _compile_model(model, section.compile_options)

        return model
