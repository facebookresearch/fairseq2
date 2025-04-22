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
from fairseq2.context import RuntimeContext
from fairseq2.error import NotSupportedError
from fairseq2.gang import GangError, Gangs
from fairseq2.logging import log
from fairseq2.models import (
    InvalidModelTypeError,
    model_asset_card_error,
    ModelConfigLoadError,
    ModelHandler,
    ModelLoadError,
    ShardedModelLoadError,
    UnknownModelError,
    UnknownModelFamilyError,
)
from fairseq2.nn.utils.module import remove_parametrizations
from fairseq2.recipes import Model, RecipeError

# isort: split

from fairseq2.recipes.common._distributed import broadcast_model
from fairseq2.recipes.common._error import ModelParallelismNotSupportedError
from fairseq2.recipes.common._model import LocalModel, maybe_compile_model
from fairseq2.recipes.config import ReferenceModelSection, TorchCompileSection
from fairseq2.recipes.utils.log import log_model
from fairseq2.registry import Provider
from fairseq2.typing import DataType
from torch.nn import Module


def setup_reference_model(
    kls: type[Module],
    context: RuntimeContext,
    model_section: ReferenceModelSection,
    gangs: Gangs,
    dtype: DataType,
    mp: bool,
    torch_compile_section: TorchCompileSection,
) -> Model:
    print("setzler in setup_reference_model")
    model = load_reference_model(kls, context, model_section, gangs, dtype, mp)
    print("here?")

    broadcast_model(model, gangs)

    model = prepare_reference_model(context, model, torch_compile_section)

    log_model(log, model.module, gangs)

    return model


def load_reference_model(
    kls: type[Module],
    context: RuntimeContext,
    model_section: ReferenceModelSection,
    gangs: Gangs,
    dtype: DataType,
    mp: bool,
) -> Model:
    model_handlers = context.get_registry(ModelHandler)

    loader = ReferenceModelLoader(kls, context.asset_store, model_handlers)

    try:
        return loader.load(model_section, gangs, dtype, mp)
    except ShardedModelLoadError:
        raise
    except ModelLoadError as ex:
        raise RecipeError(
            f"The '{ex.model_name}' model cannot be loaded. See the nested exception for details."
        ) from ex


@final
class ReferenceModelLoader:
    _kls: type[Module]
    _asset_store: AssetStore
    _model_handlers: Provider[ModelHandler]

    def __init__(
        self,
        kls: type[Module],
        asset_store: AssetStore,
        model_handlers: Provider[ModelHandler],
    ) -> None:
        self._kls = kls
        self._asset_store = asset_store
        self._model_handlers = model_handlers

    def load(
        self,
        model_section: ReferenceModelSection,
        gangs: Gangs,
        dtype: DataType,
        mp: bool,
    ) -> Model:
        model_name = model_section.name

        try:
            card = self._asset_store.retrieve_card(model_name)
        except AssetCardNotFoundError:
            raise UnknownModelError(model_name) from None
        except AssetCardError as ex:
            raise model_asset_card_error(model_name) from ex

        try:
            model_family = card.field("model_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise UnknownModelError(model_name) from None
        except AssetCardError as ex:
            raise model_asset_card_error(model_name) from ex

        try:
            handler = self._model_handlers.get(model_family)
        except LookupError:
            raise UnknownModelFamilyError(model_family, model_name) from None

        if not issubclass(handler.kls, self._kls):
            raise InvalidModelTypeError(model_name, handler.kls, self._kls) from None

        if gangs.root.size != gangs.dp.size:
            if not handler.supports_sharding:
                raise ModelParallelismNotSupportedError(model_name)

        try:
            model_config = handler.load_config(card)
        except ModelConfigLoadError as ex:
            raise ModelLoadError(
                model_name, f"The '{model_name}' model configuration cannot be loaded. See the nested exception for details."  # fmt: skip
            ) from ex

        # Load the model.
        log.info("Loading '{}' model on data parallel rank 0.", model_name)

        if mp:
            dtype = torch.float32

        try:
            if gangs.dp.rank == 0:
                module = handler.load(card, gangs, dtype, model_config)
            else:
                module = handler.create(
                    model_config, gangs, dtype, meta=handler.supports_meta
                )
        except NotSupportedError as ex:
            raise ModelLoadError(
                model_name, f"The '{model_name}' model cannot be constructed due to an unsupported operation. See the nested exception for details."  # fmt: skip
            ) from ex

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise ModelLoadError(
                model_name, f"The collective barrier after the '{model_name}' model load operation has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        module.eval()

        log.info("Model loaded on data parallel rank 0.")

        return LocalModel(model_name, module, model_config, handler)


def prepare_reference_model(
    context: RuntimeContext, model: Model, torch_compile_section: TorchCompileSection
) -> Model:
    remove_parametrizations(model.module)

    maybe_compile_model(model, torch_compile_section)

    return model
