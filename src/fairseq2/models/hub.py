# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Generic, TypeVar, cast, final

import torch
from torch import Tensor
from torch.nn import Module

from fairseq2.assets import (
    AssetCard,
    AssetCardFieldNotFoundError,
    AssetNotFoundError,
    AssetStore,
)
from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.error import ContractError
from fairseq2.gang import Gangs, create_fake_gangs
from fairseq2.models.error import (
    ModelConfigLoadError,
    ModelLoadError,
    UnknownModelError,
    UnknownModelFamilyError,
)
from fairseq2.models.handler import ModelFamilyHandler
from fairseq2.runtime.dependency import (
    DependencyNotFoundError,
    DependencyResolver,
    get_dependency_resolver,
)

ModelT = TypeVar("ModelT", bound=Module)

ModelConfigT = TypeVar("ModelConfigT")


@final
class ModelHub(Generic[ModelT, ModelConfigT]):
    _handler: ModelFamilyHandler
    _asset_store: AssetStore
    _resolver: DependencyResolver

    def __init__(
        self,
        handler: ModelFamilyHandler,
        asset_store: AssetStore,
        resolver: DependencyResolver,
    ) -> None:
        self._handler = handler
        self._asset_store = asset_store
        self._resolver = resolver

    def iter_model_cards(self) -> Iterable[AssetCard]:
        return self._asset_store.find_cards("model_family", self._handler.family)

    def get_archs(self) -> list[str]:
        return self._handler.get_archs()

    def get_arch_config(self, arch: str | None = None) -> ModelConfigT:
        config = self._handler.get_arch_config(arch)

        return cast(ModelConfigT, config)

    def load_model_config(self, name_or_card: str | AssetCard) -> ModelConfigT:
        if isinstance(name_or_card, AssetCard):
            card = name_or_card

            name = card.name
        else:
            name = name_or_card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError:
                raise UnknownModelError(name) from None

        try:
            family = card.field("model_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise UnknownModelError(name) from None

        if family != self._handler.family:
            raise ModelConfigLoadError(
                name, f"The '{name}' model does not belong to the '{family}' family."
            )

        config = self._handler.load_model_config(card)

        return cast(ModelConfigT, config)

    def create_new_model(
        self,
        config: ModelConfigT,
        *,
        gangs: Gangs | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
        meta: bool = False,
    ) -> ModelT:
        gangs = self._get_gangs(gangs, device)

        if dtype is None:
            dtype = torch.get_default_dtype()

        model = self._handler.create_new_model(
            self._resolver, config, gangs, dtype, meta
        )

        return cast(ModelT, model)

    def load_model(
        self,
        name_or_card: str | AssetCard,
        *,
        gangs: Gangs | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
        config: ModelConfigT | None = None,
        mmap: bool = False,
    ) -> ModelT:
        gangs = self._get_gangs(gangs, device)

        if isinstance(name_or_card, AssetCard):
            card = name_or_card

            name = card.name
        else:
            name = name_or_card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError:
                raise UnknownModelError(name) from None

        try:
            family = card.field("model_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise UnknownModelError(name) from None

        if family != self._handler.family:
            raise ModelLoadError(
                name, f"The '{name}' model does not belong to the '{family}' family."
            )

        if dtype is None:
            dtype = torch.get_default_dtype()

        model = self._handler.load_model(
            self._resolver, card, gangs, dtype, config=config, mmap=mmap
        )

        return cast(ModelT, model)

    def load_model_from_path(
        self,
        path: Path,
        config: ModelConfigT,
        *,
        gangs: Gangs | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
        mmap: bool = False,
        restrict: bool | None = None,
    ) -> ModelT:
        gangs = self._get_gangs(gangs, device)

        if dtype is None:
            dtype = torch.get_default_dtype()

        model = self._handler.load_model_from_path(
            self._resolver, path, config, gangs, dtype, mmap=mmap, restrict=restrict
        )

        return cast(ModelT, model)

    def iter_checkpoint(
        self,
        path: Path,
        config: ModelConfigT,
        *,
        gangs: Gangs | None = None,
        mmap: bool = False,
        restrict: bool | None = None,
    ) -> Iterable[tuple[str, Tensor]]:
        gangs = self._get_gangs(gangs, device=None)

        return self._handler.iter_checkpoint(
            path, config, gangs, mmap=mmap, restrict=restrict
        )

    def _get_gangs(self, gangs: Gangs | None, device: Device | None) -> Gangs:
        if gangs is not None and device is not None:
            raise ValueError(
                "`gangs` and `device` must not be specified at the same time."
            )

        if device is not None:
            if device.type == "meta":
                raise ValueError("`device` must be a real device.")

            return create_fake_gangs(device)

        if gangs is None:
            device = torch.get_default_device()

            return create_fake_gangs(device)

        if gangs.root.device.type == "meta":
            raise ValueError("`gangs.root` must be on a real device.")

        return gangs

    @property
    def handler(self) -> ModelFamilyHandler:
        return self._handler


@final
class ModelHubAccessor(Generic[ModelT, ModelConfigT]):
    _family: str
    _kls: type[ModelT]
    _config_kls: type[ModelConfigT]

    def __init__(
        self, family: str, kls: type[ModelT], config_kls: type[ModelConfigT]
    ) -> None:
        self._family = family
        self._kls = kls
        self._config_kls = config_kls

    def __call__(self) -> ModelHub[ModelT, ModelConfigT]:
        resolver = get_dependency_resolver()

        asset_store = resolver.resolve(AssetStore)

        handlers = resolver.get_provider(ModelFamilyHandler)

        family = self._family

        try:
            handler = handlers.get(family)
        except DependencyNotFoundError:
            raise UnknownModelFamilyError(family) from None

        if not issubclass(handler.kls, self._kls):
            raise ContractError(
                f"`kls` is `{self._kls}`, but the type of the '{family}' model family is `{handler.kls}`."
            )

        if not issubclass(handler.config_kls, self._config_kls):
            raise ContractError(
                f"`config_kls` is `{self._config_kls}`, but the configuration type of the '{family}' model family is `{handler.config_kls}`."
            )

        return ModelHub(handler, asset_store, resolver)
