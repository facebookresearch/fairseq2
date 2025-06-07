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
from fairseq2.data_type import DataType
from fairseq2.dependency import DependencyResolver
from fairseq2.device import Device
from fairseq2.gang import Gangs, fake_gangs
from fairseq2.models.error import (
    InvalidModelConfigTypeError,
    InvalidModelTypeError,
    ModelConfigLoadError,
    UnknownModelError,
    UnknownModelFamilyError,
    model_asset_card_error,
)
from fairseq2.models.handler import ModelHandler
from fairseq2.typing import Provider

ModelT = TypeVar("ModelT", bound=Module)

ModelConfigT = TypeVar("ModelConfigT")


@final
class ModelHub(Generic[ModelT, ModelConfigT]):
    _kls: type[ModelT]
    _config_kls: type[ModelConfigT]
    _asset_store: AssetStore
    _handlers: Provider[ModelHandler]
    _resolver: DependencyResolver

    def __init__(
        self,
        kls: type[ModelT],
        config_kls: type[ModelConfigT],
        asset_store: AssetStore,
        handlers: Provider[ModelHandler],
        resolver: DependencyResolver,
    ) -> None:
        self._kls = kls
        self._config_kls = config_kls
        self._asset_store = asset_store
        self._handlers = handlers
        self._resolver = resolver

    def load_config(self, name_or_card: str | AssetCard) -> ModelConfigT:
        def asset_card_error(name: str) -> Exception:
            return ModelConfigLoadError(
                name, f"The '{name}' asset card cannot be read. See the nested exception for details."  # fmt: skip
            )

        if isinstance(name_or_card, AssetCard):
            card = name_or_card

            name = card.name
        else:
            name = name_or_card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetCardNotFoundError:
                raise UnknownModelError(name) from None
            except AssetCardError as ex:
                raise asset_card_error(name) from ex

        try:
            family = card.field("model_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise UnknownModelError(name) from None
        except AssetCardError as ex:
            raise asset_card_error(name) from ex

        try:
            handler = self._handlers.resolve(family)
        except LookupError:
            raise UnknownModelFamilyError(family, name) from None

        if not issubclass(handler.config_kls, self._config_kls):
            raise InvalidModelConfigTypeError(
                name, handler.config_kls, self._config_kls
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
        mmap: bool = False,
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
                raise ValueError("`gangs.root` must be on a real device.")

        if isinstance(name_or_card, AssetCard):
            card = name_or_card

            name = card.name
        else:
            name = name_or_card

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

        if not issubclass(handler.kls, self._kls):
            raise InvalidModelTypeError(name, handler.kls, self._kls)

        if dtype is None:
            dtype = torch.get_default_dtype()

        model = handler.load(self._resolver, card, gangs, dtype, config, mmap=mmap)

        return cast(ModelT, model)


@final
class ModelHubAccessor(Generic[ModelT, ModelConfigT]):
    _kls: type[ModelT]
    _config_kls: type[ModelConfigT]

    def __init__(self, kls: type[ModelT], config_kls: type[ModelConfigT]) -> None:
        self._kls = kls
        self._config_kls = config_kls

    def __call__(self) -> ModelHub[ModelT, ModelConfigT]:
        from fairseq2 import get_dependency_resolver

        resolver = get_dependency_resolver()

        asset_store = resolver.resolve(AssetStore)

        handlers = resolver.resolve_provider(ModelHandler)

        return ModelHub(self._kls, self._config_kls, asset_store, handlers, resolver)
