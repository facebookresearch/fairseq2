# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator
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
from fairseq2.models.family import ModelFamily
from fairseq2.runtime.dependency import get_dependency_resolver
from fairseq2.runtime.lookup import Lookup
from fairseq2.utils.warn import _warn_deprecated

ModelT = TypeVar("ModelT", bound=Module)

ModelConfigT = TypeVar("ModelConfigT")


@final
class ModelHub(Generic[ModelT, ModelConfigT]):
    def __init__(self, family: ModelFamily, asset_store: AssetStore) -> None:
        self._family = family
        self._asset_store = asset_store

    def iter_cards(self) -> Iterator[AssetCard]:
        return self._asset_store.find_cards("model_family", self._family.name)

    def get_archs(self) -> set[str]:
        return self._family.get_archs()

    def get_arch_config(self, arch: str) -> ModelConfigT:
        config = self.maybe_get_arch_config(arch)
        if config is None:
            raise ModelArchitectureNotKnownError(arch, self._family.name)

        return config

    def maybe_get_arch_config(self, arch: str) -> ModelConfigT | None:
        config = self._family.maybe_get_arch_config(arch)

        return cast(ModelConfigT | None, config)

    def get_model_config(self, card: AssetCard | str) -> ModelConfigT:
        if isinstance(card, str):
            name = card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError:
                raise ModelNotKnownError(name) from None
        else:
            name = card.name

        family_name = card.field("model_family").as_(str)

        if family_name != self._family.name:
            msg = f"family field of the {name} asset card is expected to be {self._family.name}, but is {family_name} instead."

            raise AssetCardError(name, msg)

        config = self._family.get_model_config(card)

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

        model = self._family.create_new_model(config, gangs, dtype, meta)

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
        progress: bool = True,
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

        family_name = card.field("model_family").as_(str)

        if family_name != self._family.name:
            msg = f"family field of the {name} asset card is expected to be {self._family.name}, but is {family_name} instead."

            raise AssetCardError(name, msg)

        if dtype is None:
            dtype = torch.get_default_dtype()

        model = self._family.load_model(card, gangs, dtype, config, mmap, progress)

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
        progress: bool = True,
    ) -> ModelT:
        gangs = _get_effective_gangs(gangs, device)

        if dtype is None:
            dtype = torch.get_default_dtype()

        model = self._family.load_custom_model(
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
    ) -> Iterator[tuple[str, Tensor]]:
        gangs = _get_effective_gangs(gangs, device=None)

        return self._family.iter_checkpoint(path, config, gangs, mmap, restrict)


@final
class ModelHubAccessor(Generic[ModelT, ModelConfigT]):
    def __init__(
        self, family_name: str, kls: type[ModelT], config_kls: type[ModelConfigT]
    ) -> None:
        self._family_name = family_name
        self._kls = kls
        self._config_kls = config_kls

    def __call__(self) -> ModelHub[ModelT, ModelConfigT]:
        resolver = get_dependency_resolver()

        asset_store = resolver.resolve(AssetStore)

        name = self._family_name

        family = resolver.resolve_optional(ModelFamily, key=name)
        if family is None:
            raise ModelFamilyNotKnownError(name)

        if not issubclass(family.kls, self._kls):
            raise InternalError(
                f"`kls` is `{self._kls}`, but the type of the {name} model family is `{family.kls}`."
            )

        if not issubclass(family.config_kls, self._config_kls):
            raise InternalError(
                f"`config_kls` is `{self._config_kls}`, but the configuration type of the {name} model family is `{family.config_kls}`."
            )

        return ModelHub(family, asset_store)


class ModelNotKnownError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is not a known model.")

        self.name = name


class ModelFamilyNotKnownError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is not a known model family.")

        self.name = name


class ModelArchitectureNotKnownError(Exception):
    def __init__(self, arch: str, family: str | None = None) -> None:
        """
        ``family`` defaults to ``None`` due to backwards-compatibility. New code
        must specify a model family when raising this error.
        """
        if family is None:
            _warn_deprecated(
                "`ModelArchitectureNotKnownError` will require a `family` argument starting fairseq2 v0.12."
            )

            super().__init__(f"{arch} is not a known model architecture.")
        else:
            super().__init__(f"{arch} is not a known {family} model architecture.")

        self.arch = arch
        self.family = family


def load_model(
    card: AssetCard | str,
    *,
    gangs: Gangs | None = None,
    device: Device | None = None,
    dtype: DataType | None = None,
    config: object = None,
    mmap: bool = False,
    progress: bool = True,
) -> Module:
    resolver = get_dependency_resolver()

    global_loader = resolver.resolve(GlobalModelLoader)

    return global_loader.load(card, gangs, device, dtype, config, mmap, progress)


@final
class GlobalModelLoader:
    def __init__(self, asset_store: AssetStore, families: Lookup[ModelFamily]) -> None:
        self._asset_store = asset_store
        self._families = families

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

        family_name = card.field("model_family").as_(str)

        family = self._families.maybe_get(family_name)
        if family is None:
            msg = f"family field of the {name} asset card is expected to be a supported model family, but is {family_name} instead."

            raise AssetCardError(name, msg)

        if dtype is None:
            dtype = torch.get_default_dtype()

        return family.load_model(card, gangs, dtype, config, mmap, progress)


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
