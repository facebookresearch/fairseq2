# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Generic, TypeVar, cast, final

import torch
from torch.nn import Module

from fairseq2.assets import AssetCard, AssetStore
from fairseq2.context import get_runtime_context
from fairseq2.gang import Gangs, fake_gangs
from fairseq2.models._handler import ModelHandler, ModelNotFoundError, get_model_family
from fairseq2.registry import Provider
from fairseq2.typing import CPU, DataType, Device

ModelT = TypeVar("ModelT", bound=Module)

ModelConfigT = TypeVar("ModelConfigT")


@final
class ModelHub(Generic[ModelT, ModelConfigT]):
    _kls: type[ModelT]
    _config_kls: type[ModelConfigT]
    _asset_store: AssetStore
    _model_handlers: Provider[ModelHandler]

    def __init__(
        self,
        kls: type[ModelT],
        config_kls: type[ModelConfigT],
        asset_store: AssetStore,
        model_handlers: Provider[ModelHandler],
    ) -> None:
        self._kls = kls
        self._config_kls = config_kls
        self._asset_store = asset_store
        self._model_handlers = model_handlers

    def load_config(self, name_or_card: str | AssetCard) -> ModelConfigT:
        if isinstance(name_or_card, AssetCard):
            card = name_or_card
        else:
            card = self._asset_store.retrieve_card(name_or_card)

        family = get_model_family(card)

        try:
            handler = self._model_handlers.get(family)
        except LookupError:
            raise ModelNotFoundError(card.name) from None

        if not issubclass(handler.config_kls, self._config_kls):
            raise TypeError(
                f"The '{card.name}' model configuration is expected to be of type `{self._config_kls}`, but is of type `{handler.config_kls}` instead."
            )

        config = handler.load_config(card)

        return cast(ModelConfigT, config)

    def load(
        self,
        name_or_card: str | AssetCard,
        *,
        gangs: Gangs | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
        config: ModelConfigT | None = None,
    ) -> ModelT:
        if gangs is not None and device is not None:
            raise ValueError(
                "`gangs` and `device` must not be specified at the same time."
            )

        if device is not None:
            if device.type == "meta":
                raise ValueError("`device` must be a real device.")

            gangs = fake_gangs(device)
        elif gangs is None:
            gangs = fake_gangs(CPU)
        else:
            if gangs.root.device.type == "meta":
                raise ValueError("`gangs` must be on a real device.")

        if isinstance(name_or_card, AssetCard):
            card = name_or_card
        else:
            card = self._asset_store.retrieve_card(name_or_card)

        family = get_model_family(card)

        try:
            handler = self._model_handlers.get(family)
        except LookupError:
            raise ModelNotFoundError(card.name) from None

        if not issubclass(handler.kls, self._kls):
            raise TypeError(
                f"The '{card.name}' model is expected to be of type `{self._kls}`, but is of type `{handler.kls}` instead."
            )

        if dtype is None:
            dtype = torch.float32

        model = handler.load(card, gangs, dtype, config=config)

        return cast(ModelT, model)


@final
class ModelHubAccessor(Generic[ModelT, ModelConfigT]):
    _kls: type[ModelT]
    _config_kls: type[ModelConfigT]

    def __init__(self, kls: type[ModelT], config_kls: type[ModelConfigT]) -> None:
        self._kls = kls
        self._config_kls = config_kls

    def __call__(self) -> ModelHub[ModelT, ModelConfigT]:
        context = get_runtime_context()

        model_handlers = context.get_registry(ModelHandler)

        return ModelHub(
            self._kls, self._config_kls, context.asset_store, model_handlers
        )
