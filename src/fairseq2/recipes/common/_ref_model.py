# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Generic, TypeVar, cast, final

import torch
from torch.nn import Module

from fairseq2.assets import (
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetCardNotFoundError,
    AssetStore,
)
from fairseq2.context import RuntimeContext
from fairseq2.error import NotSupportedError, ProgramError
from fairseq2.gang import GangError, Gangs
from fairseq2.logging import log
from fairseq2.models import (
    InvalidModelTypeError,
    ModelConfigLoadError,
    ModelHandler,
    ModelLoadError,
    ShardedModelLoadError,
    UnknownModelError,
    UnknownModelFamilyError,
    model_asset_card_error,
)
from fairseq2.models.compile import compile_model
from fairseq2.nn.utils.module import remove_parametrizations
from fairseq2.recipes.common._distributed import broadcast_model
from fairseq2.recipes.common._model import ModelParallelismNotSupportedError
from fairseq2.recipes.utils.log import log_model
from fairseq2.registry import Provider
from fairseq2.typing import DataType

ModelT = TypeVar("ModelT", bound=Module)


def setup_reference_model(
    kls: type[ModelT],
    context: RuntimeContext,
    model_name: str,
    gangs: Gangs,
    dtype: DataType,
    mp: bool,
    torch_compile: bool,
) -> ModelT:
    model = load_reference_model(kls, context, model_name, gangs, dtype, mp)

    broadcast_model(model_name, model, gangs)

    model = prepare_reference_model(context, model_name, model, gangs, torch_compile)

    log_model(log, model, gangs)

    return model


def load_reference_model(
    kls: type[ModelT],
    context: RuntimeContext,
    model_name: str,
    gangs: Gangs,
    dtype: DataType,
    mp: bool,
) -> ModelT:
    model_handlers = context.get_registry(ModelHandler)

    loader = ReferenceModelLoader(kls, context.asset_store, model_handlers)

    try:
        return loader.load(model_name, gangs, dtype, mp)
    except ShardedModelLoadError:
        raise
    except ModelLoadError as ex:
        raise ProgramError(
            f"The '{ex.model_name}' model cannot be loaded. See the nested exception for details."
        ) from ex


@final
class ReferenceModelLoader(Generic[ModelT]):
    _kls: type[ModelT]
    _asset_store: AssetStore
    _model_handlers: Provider[ModelHandler]

    def __init__(
        self,
        kls: type[ModelT],
        asset_store: AssetStore,
        model_handlers: Provider[ModelHandler],
    ) -> None:
        self._kls = kls
        self._asset_store = asset_store
        self._model_handlers = model_handlers

    def load(self, model_name: str, gangs: Gangs, dtype: DataType, mp: bool) -> ModelT:
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

        if not handler.supports_sharding and gangs.root.size != gangs.dp.size:
            raise ModelParallelismNotSupportedError(model_family, model_name)

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
                model = handler.load(card, gangs, dtype, model_config)
            else:
                model = handler.create(
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
                model_name, f"The collective barrier after the load of the '{model_name}' model has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        model.eval()

        log.info("Model loaded on data parallel rank 0.")

        return cast(ModelT, model)


def prepare_reference_model(
    context: RuntimeContext,
    model_name: str,
    model: ModelT,
    gangs: Gangs,
    torch_compile: bool,
) -> ModelT:
    remove_parametrizations(model)

    if torch_compile:
        log.info("Compiling '{}' model.", model_name)

        model = cast(ModelT, compile_model(model, gangs))

        log.info("Model compiled.")

    return model
