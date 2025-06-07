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

from fairseq2.assets import AssetCard, AssetCardError, AssetNotFoundError, AssetStore
from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.error import InternalError
from fairseq2.gang import Gangs, create_fake_gangs
from fairseq2.models.handler import ModelFamilyHandler
from fairseq2.runtime.dependency import get_dependency_resolver
from fairseq2.runtime.provider import Provider

ModelT = TypeVar("ModelT", bound=Module)

ModelConfigT = TypeVar("ModelConfigT")


@final
class ModelHub(Generic[ModelT, ModelConfigT]):
    def __init__(self, handler: ModelFamilyHandler, asset_store: AssetStore) -> None:
        self._handler = handler
        self._asset_store = asset_store

    def iter_cards(self) -> Iterable[AssetCard]:
        return self._asset_store.find_cards("model_family", self._handler.family)

    def get_archs(self) -> set[str]:
        return self._handler.get_archs()

    def get_arch_config(self, arch: str) -> ModelConfigT:
        config = self._handler.get_arch_config(arch)

        return cast(ModelConfigT, config)

    def get_model_config(self, card: AssetCard | str) -> ModelConfigT:
        if isinstance(card, str):
            name = card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError:
                raise ModelNotKnownError(name) from None
        else:
            name = card.name

        family = card.field("model_family").as_(str)

        if family != self._handler.family:
            msg = f"family field of the {name} asset card is expected to be '{self._handler.family}', but is '{family}' instead."

            raise AssetCardError(name, msg)

        config = self._handler.get_model_config(card)

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
        gangs = _get_effective_gangs(gangs, device)

        if dtype is None:
            dtype = torch.get_default_dtype()

        model = self._handler.create_new_model(config, gangs, dtype, meta)

        return cast(ModelT, model)

    def load_model(
        self,
        card: AssetCard | str,
        *,
        gangs: Gangs | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
        config: ModelConfigT | None = None,
        mmap: bool = False,
        progress: bool = False,
    ) -> ModelT:
        gangs = _get_effective_gangs(gangs, device)

        if isinstance(card, str):
            name = card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError:
                raise ModelNotKnownError(name) from None
        else:
            name = card.name

        family = card.field("model_family").as_(str)

        if family != self._handler.family:
            msg = f"family field of the {name} asset card is expected to be {self._handler.family}, but is {family} instead."

            raise AssetCardError(name, msg)

        if dtype is None:
            dtype = torch.get_default_dtype()

        model = self._handler.load_model(card, gangs, dtype, config, mmap, progress)

        return cast(ModelT, model)

    def load_custom_model(
        self,
        path: Path,
        config: ModelConfigT,
        *,
        gangs: Gangs | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
        mmap: bool = False,
        restrict: bool | None = None,
        progress: bool = False,
    ) -> ModelT:
        gangs = _get_effective_gangs(gangs, device)

        if dtype is None:
            dtype = torch.get_default_dtype()

        model = self._handler.load_custom_model(
            path, config, gangs, dtype, mmap, restrict, progress
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
        gangs = _get_effective_gangs(gangs, device=None)

        return self._handler.iter_checkpoint(path, config, gangs, mmap, restrict)

    @property
    def handler(self) -> ModelFamilyHandler:
        return self._handler


@final
class ModelHubAccessor(Generic[ModelT, ModelConfigT]):
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
        except LookupError:
            raise ModelFamilyNotKnownError(family) from None

        if not issubclass(handler.kls, self._kls):
            raise InternalError(
                f"`kls` is `{self._kls}`, but the type of the {family} model family is `{handler.kls}`."
            )

        if not issubclass(handler.config_kls, self._config_kls):
            raise InternalError(
                f"`config_kls` is `{self._config_kls}`, but the configuration type of the {family} model family is `{handler.config_kls}`."
            )

        return ModelHub(handler, asset_store)


class ModelNotKnownError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is not a known model.")

        self.name = name


class ModelFamilyNotKnownError(Exception):
    def __init__(self, family: str) -> None:
        super().__init__(f"{family} is not a known model family.")

        self.family = family


def load_model(
    card: AssetCard | str,
    *,
    gangs: Gangs | None = None,
    device: Device | None = None,
    dtype: DataType | None = None,
    config: object = None,
    mmap: bool = False,
    progress: bool = False,
) -> Module:
    resolver = get_dependency_resolver()

    asset_store = resolver.resolve(AssetStore)

    handlers = resolver.get_provider(ModelFamilyHandler)

    loader = GlobalModelLoader(asset_store, handlers)

    return loader.load(card, gangs, device, dtype, config, mmap, progress)


@final
class GlobalModelLoader:
    def __init__(
        self, asset_store: AssetStore, handlers: Provider[ModelFamilyHandler]
    ) -> None:
        self._asset_store = asset_store
        self._handlers = handlers

    def load(
        self,
        card: AssetCard | str,
        gangs: Gangs | None,
        device: Device | None,
        dtype: DataType | None,
        config: object | None,
        mmap: bool,
        progress: bool,
    ) -> Module:
        gangs = _get_effective_gangs(gangs, device)

        if isinstance(card, str):
            name = card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError:
                raise ModelNotKnownError(name) from None
        else:
            name = card.name

        family = card.field("model_family").as_(str)

        handler = self._handlers.maybe_get(family)
        if handler is None:
            msg = f"family field of the {name} asset card is expected to be a supported model family, but is '{family}' instead."

            raise AssetCardError(name, msg)

        if dtype is None:
            dtype = torch.get_default_dtype()

        return handler.load_model(card, gangs, dtype, config, mmap, progress)


def _get_effective_gangs(gangs: Gangs | None, device: Device | None) -> Gangs:
    if gangs is not None and device is not None:
        raise ValueError("`gangs` and `device` must not be specified at the same time.")

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
