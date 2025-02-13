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
    AssetCard,
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetCardNotFoundError,
    AssetStore,
)
from fairseq2.context import get_runtime_context
from fairseq2.gang import Gangs, fake_gangs
from fairseq2.models._error import (
    InvalidModelConfigTypeError,
    InvalidModelTypeError,
    ModelConfigLoadError,
    UnknownModelError,
    UnknownModelFamilyError,
    model_asset_card_error,
)
from fairseq2.models._handler import ModelHandler
from fairseq2.registry import Provider
from fairseq2.typing import DataType, Device

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
        def asset_card_error(model_name: str) -> Exception:
            return ModelConfigLoadError(
                model_name, f"The '{model_name}' asset card cannot be read. See the nested exception for details."  # fmt: skip
            )

        if isinstance(name_or_card, AssetCard):
            card = name_or_card

            model_name = card.name
        else:
            model_name = name_or_card

            try:
                card = self._asset_store.retrieve_card(model_name)
            except AssetCardNotFoundError:
                raise UnknownModelError(model_name) from None
            except AssetCardError as ex:
                raise asset_card_error(model_name) from ex

        try:
            model_family = card.field("model_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise UnknownModelError(model_name) from None
        except AssetCardError as ex:
            raise asset_card_error(model_name) from ex

        try:
            handler = self._model_handlers.get(model_family)
        except LookupError:
            raise UnknownModelFamilyError(model_family, model_name) from None

        if not issubclass(handler.config_kls, self._config_kls):
            raise InvalidModelConfigTypeError(
                model_name, handler.config_kls, self._config_kls
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
            device = torch.get_default_device()

            gangs = fake_gangs(device)
        else:
            if gangs.root.device.type == "meta":
                raise ValueError("`gangs` must be on a real device.")

        if isinstance(name_or_card, AssetCard):
            card = name_or_card

            model_name = card.name
        else:
            model_name = name_or_card

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
            raise InvalidModelTypeError(model_name, handler.kls, self._kls)

        if dtype is None:
            dtype = torch.get_default_dtype()

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
