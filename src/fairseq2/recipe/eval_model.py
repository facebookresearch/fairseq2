# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import final

import torch
from typing_extensions import override

from fairseq2.assets import AssetCardError, AssetStore
from fairseq2.data_type import DataType
from fairseq2.error import InternalError, raise_operational_system_error
from fairseq2.gang import GangError, Gangs, raise_operational_gang_error
from fairseq2.logging import log
from fairseq2.model import Model, StandardModel
from fairseq2.models import (
    ModelFamilyHandler,
    ModelFamilyNotKnownError,
    ModelNotKnownError,
)
from fairseq2.nn.utils.module import broadcast_module, remove_parametrizations
from fairseq2.recipe.asset_config import AssetConfigOverrider
from fairseq2.recipe.base import Recipe, RecipeContext
from fairseq2.recipe.compile import _compile_model
from fairseq2.recipe.config import (
    ReferenceModelSection,
    TrainerSection,
    get_config_section,
)
from fairseq2.recipe.model import (
    ModelArchitectureNotKnownError,
    ModelCheckpointNotFoundError,
    ModelParallelismNotSupportedError,
)
from fairseq2.runtime.config_registry import ConfigNotFoundError
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.runtime.provider import Provider
from fairseq2.utils.log import log_model


def load_reference_model(resolver: DependencyResolver, section_name: str) -> Model:
    trainer_section = get_config_section(resolver, "trainer", TrainerSection)

    mp = trainer_section.mixed_precision != "off"

    return load_eval_model(resolver, section_name, trainer_section.dtype, mp)


def load_eval_model(
    resolver: DependencyResolver, section_name: str, dtype: DataType, mp: bool
) -> Model:
    section = get_config_section(resolver, section_name, ReferenceModelSection)

    factory = resolver.resolve(EvalModelFactory)

    return factory.create(section_name, section, dtype, mp)


@final
class EvalModelFactory:
    def __init__(
        self,
        bootstrapper: EvalModelBootstrapper,
        preparer: EvalModelPreparer,
        gangs: Gangs,
    ) -> None:
        self._bootstrapper = bootstrapper
        self._preparer = preparer
        self._gangs = gangs

    def create(
        self,
        section_name: str,
        section: ReferenceModelSection,
        dtype: DataType,
        mp: bool,
    ) -> Model:
        model = self._bootstrapper.bootstrap(section_name, section, dtype, mp)

        model = self._preparer.prepare(model, section_name, section)

        log_model(model.module, self._gangs)

        return model


class EvalModelBootstrapper(ABC):
    @abstractmethod
    def bootstrap(
        self,
        section_name: str,
        section: ReferenceModelSection,
        dtype: DataType,
        mp: bool,
    ) -> Model: ...


@final
class StandardEvalModelBootstrapper(EvalModelBootstrapper):
    def __init__(
        self,
        handlers: Provider[ModelFamilyHandler],
        asset_store: AssetStore,
        asset_config_overrider: AssetConfigOverrider,
        gangs: Gangs,
    ) -> None:
        self._handlers = handlers
        self._asset_store = asset_store
        self._asset_config_overrider = asset_config_overrider
        self._gangs = gangs

    @override
    def bootstrap(
        self,
        section_name: str,
        section: ReferenceModelSection,
        dtype: DataType,
        mp: bool,
    ) -> Model:
        if section.path is not None:
            return self._load_model_from_path(section_name, section, dtype, mp)

        if section.name is not None:
            return self._load_model_from_card(section_name, section, dtype, mp)

        raise InternalError("`section.name` and `section.family` are both `None`.")

    def _load_model_from_card(
        self,
        section_name: str,
        section: ReferenceModelSection,
        dtype: DataType,
        mp: bool,
    ) -> Model:
        name = section.name
        if name is None:
            raise InternalError("`section.name` is `None`.")

        card = self._asset_store.maybe_retrieve_card(name)
        if card is None:
            raise ModelNotKnownError(name)

        family = card.field("model_family").as_(str)

        handler = self._handlers.maybe_get(family)
        if handler is None:
            msg = f"family field of the {name} asset card is expected to be a supported model family, but is '{family}' instead."

            raise AssetCardError(name, msg)

        config = handler.get_model_config(card)

        if section.config_overrides is not None:
            config = self._asset_config_overrider.apply_overrides(
                section_name, config, section.config_overrides
            )

        gangs = self._gangs

        if gangs.root.size != gangs.dp.size:
            if not handler.supports_model_parallelism:
                raise ModelParallelismNotSupportedError(name)

        # Load the model.
        if mp:
            dtype = torch.float32

        log.info("Loading {} model on data parallel rank 0.", name)

        if gangs.dp.rank == 0:
            module = handler.load_model(
                card, gangs, dtype, config, section.mmap, progress=True
            )
        else:
            module = handler.create_new_model(
                config, gangs, dtype, meta=handler.supports_meta
            )

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

        log.info("Model loaded on data parallel rank 0.")

        module.eval()

        return StandardModel(name, module, config, handler)

    def _load_model_from_path(
        self,
        section_name: str,
        section: ReferenceModelSection,
        dtype: DataType,
        mp: bool,
    ) -> Model:
        path = section.path
        if path is None:
            raise InternalError("`section.path` is `None`.")

        family = section.family
        if family is None:
            raise InternalError("`section.family` is `None`.")

        name = path.name

        handler = self._handlers.maybe_get(family)
        if handler is None:
            raise ModelFamilyNotKnownError(family)

        arch = section.arch
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

        if section.config_overrides is not None:
            config = self._asset_config_overrider.apply_overrides(
                section_name, config, section.config_overrides
            )

        gangs = self._gangs

        if gangs.root.size != gangs.dp.size:
            if not handler.supports_model_parallelism:
                raise ModelParallelismNotSupportedError(name)

        # Load the model.
        if mp:
            dtype = torch.float32

        log.info("Loading {} model on data parallel rank 0.", name)

        if gangs.dp.rank == 0:
            try:
                module = handler.load_custom_model(
                    path,
                    config,
                    gangs,
                    dtype,
                    section.mmap,
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

        log.info("Model loaded on data parallel rank 0.")

        module.eval()

        return StandardModel(name, module, config, handler)


class EvalModelPreparer(ABC):
    @abstractmethod
    def prepare(
        self, model: Model, section_name: str, section: ReferenceModelSection
    ) -> Model: ...


@final
class DelegatingEvalModelPreparer(EvalModelPreparer):
    def __init__(self, preparers: Iterable[EvalModelPreparer]) -> None:
        self._preparers = list(preparers)

    @override
    def prepare(
        self, model: Model, section_name: str, section: ReferenceModelSection
    ) -> Model:
        for preparer in self._preparers:
            model = preparer.prepare(model, section_name, section)

        return model


@final
class StandardEvalModelPreparer(EvalModelPreparer):
    def __init__(self, gangs: Gangs) -> None:
        self._gangs = gangs

    @override
    def prepare(
        self, model: Model, section_name: str, section: ReferenceModelSection
    ) -> Model:
        if self._gangs.dp.size > 1:
            log.info("Broadcasting {} model to all processes.", model.name)

            try:
                broadcast_module(model.module, self._gangs.dp)
            except GangError as ex:
                raise_operational_gang_error(ex)

            log.info("Model broadcasted.")

        remove_parametrizations(model.module)

        if section.compile:
            _compile_model(model, section.compile_options)

        return model


@final
class RecipeEvalModelPreparer(EvalModelPreparer):
    def __init__(self, recipe: Recipe, resolver: DependencyResolver) -> None:
        self._recipe = recipe
        self._resolver = resolver

    @override
    def prepare(
        self, model: Model, section_name: str, section: ReferenceModelSection
    ) -> Model:
        context = RecipeContext(self._resolver)

        if section_name == "model":
            return self._recipe.prepare_model(context, model)

        return self._recipe.prepare_eval_model(context, model, section_name, section)
