# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

import torch

from fairseq2.assets import (
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetCardNotFoundError,
    AssetStore,
)
from fairseq2.data_type import DataType
from fairseq2.dependency import DependencyResolver
from fairseq2.error import NotSupportedError, SetupError
from fairseq2.gang import GangError, Gangs
from fairseq2.logging import log
from fairseq2.models import (
    ModelConfigLoadError,
    ModelHandler,
    ModelLoadError,
    UnknownModelError,
    UnknownModelFamilyError,
    model_asset_card_error,
)
from fairseq2.nn.utils.module import remove_parametrizations
from fairseq2.recipe.base_model import _BaseModel
from fairseq2.recipe.compile import compile_model
from fairseq2.recipe.config import (
    EvaluatorSection,
    GeneratorSection,
    ReferenceModelSection,
    TrainerSection,
    get_config_section,
)
from fairseq2.recipe.data_parallel_model import broadcast_model
from fairseq2.recipe.error import ModelParallelismNotSupportedError
from fairseq2.recipe.model import Model
from fairseq2.recipe.utils.log import log_model
from fairseq2.typing import Provider


def setup_eval_model(resolver: DependencyResolver) -> Model:
    model = resolver.resolve(Model, key="base")

    model_section = get_config_section(resolver, "model", ReferenceModelSection)

    return _do_setup_reference_model(resolver, model, model_section)


def setup_generator_model(resolver: DependencyResolver) -> Model:
    model = resolver.resolve(Model, key="base")

    model_section = get_config_section(resolver, "model", ReferenceModelSection)

    return _do_setup_reference_model(resolver, model, model_section)


def setup_reference_model(
    resolver: DependencyResolver, model_section: ReferenceModelSection
) -> Model:
    model = load_base_reference_model(resolver, model_section)

    return _do_setup_reference_model(resolver, model, model_section)


def _do_setup_reference_model(
    resolver: DependencyResolver, model: Model, model_section: ReferenceModelSection
) -> Model:
    gangs = resolver.resolve(Gangs)

    broadcast_model(model, gangs)

    prepare_reference_model(resolver, model, model_section)

    log_model(log, model.module, gangs)

    return model


def load_base_eval_model(resolver: DependencyResolver) -> Model:
    model_section = get_config_section(resolver, "model", ReferenceModelSection)

    evaluator_section = get_config_section(resolver, "evaluator", EvaluatorSection)

    return _do_load_reference_model(
        resolver, model_section, evaluator_section.dtype, evaluator_section.amp
    )


def load_base_generator_model(resolver: DependencyResolver) -> Model:
    model_section = get_config_section(resolver, "model", ReferenceModelSection)

    generator_section = get_config_section(resolver, "generator", GeneratorSection)

    return _do_load_reference_model(
        resolver, model_section, generator_section.dtype, generator_section.amp
    )


def load_base_reference_model(
    resolver: DependencyResolver, model_section: ReferenceModelSection
) -> Model:
    trainer_section = get_config_section(resolver, "trainer", TrainerSection)

    mp = trainer_section.mixed_precision != "off"

    return _do_load_reference_model(resolver, model_section, trainer_section.dtype, mp)


def _do_load_reference_model(
    resolver: DependencyResolver,
    model_section: ReferenceModelSection,
    dtype: DataType,
    mp: bool,
) -> Model:
    asset_store = resolver.resolve(AssetStore)

    handlers = resolver.resolve_provider(ModelHandler)

    gangs = resolver.resolve(Gangs)

    loader = _ReferenceModelLoader(asset_store, handlers)

    try:
        return loader.load(resolver, model_section, gangs, dtype, mp)
    except ModelLoadError as ex:
        raise SetupError(
            f"The '{ex.model_name}' model cannot be loaded. See the nested exception for details."
        ) from ex


@final
class _ReferenceModelLoader:
    _asset_store: AssetStore
    _handlers: Provider[ModelHandler]

    def __init__(
        self, asset_store: AssetStore, handlers: Provider[ModelHandler]
    ) -> None:
        self._asset_store = asset_store
        self._handlers = handlers

    def load(
        self,
        resolver: DependencyResolver,
        model_section: ReferenceModelSection,
        gangs: Gangs,
        dtype: DataType,
        mp: bool,
    ) -> Model:
        name = model_section.name

        try:
            card = self._asset_store.retrieve_card(name)
        except AssetCardNotFoundError:
            raise UnknownModelError(name) from None
        except AssetCardError as ex:
            raise model_asset_card_error(name) from ex

        try:
            family = card.field("model_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise UnknownModelError(name) from None
        except AssetCardError as ex:
            raise model_asset_card_error(name) from ex

        try:
            handler = self._handlers.resolve(family)
        except LookupError:
            raise UnknownModelFamilyError(family, name) from None

        if gangs.root.size != gangs.dp.size:
            if not handler.supports_model_parallelism:
                raise ModelParallelismNotSupportedError(name)

        try:
            config = handler.load_config(card)
        except ModelConfigLoadError as ex:
            raise ModelLoadError(
                name, f"The '{name}' model configuration cannot be loaded. See the nested exception for details."  # fmt: skip
            ) from ex

        # Load the model.
        log.info("Loading '{}' model on data parallel rank 0.", name)

        if mp:
            dtype = torch.float32

        try:
            if gangs.dp.rank == 0:
                module = handler.load(
                    resolver, card, gangs, dtype, config, mmap=model_section.mmap
                )
            else:
                module = handler.create(
                    resolver, config, gangs, dtype, meta=handler.supports_meta
                )
        except NotSupportedError as ex:
            raise ModelLoadError(
                name, f"The '{name}' model cannot be constructed due to an unsupported operation. See the nested exception for details."  # fmt: skip
            ) from ex

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise ModelLoadError(
                name, f"The collective barrier after the '{name}' model load operation has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        module.eval()

        log.info("Model loaded on data parallel rank 0.")

        return _BaseModel(name, module, config, handler)


def prepare_reference_model(
    resolver: DependencyResolver, model: Model, model_section: ReferenceModelSection
) -> Model:
    remove_parametrizations(model.module)

    if model_section.compile:
        compile_model(model, model_section.compile_options)

    return model
