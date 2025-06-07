# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, TypeVar, cast, final

from fairseq2.assets import AssetCard, AssetCardError, AssetNotFoundError, AssetStore
from fairseq2.datasets.handler import DatasetFamilyHandler
from fairseq2.error import InternalError
from fairseq2.runtime.dependency import (
    DependencyNotFoundError,
    DependencyResolver,
    get_dependency_resolver,
)

DatasetT = TypeVar("DatasetT")

DatasetConfigT = TypeVar("DatasetConfigT")


@final
class DatasetHub(Generic[DatasetT, DatasetConfigT]):
    _handler: DatasetFamilyHandler
    _asset_store: AssetStore

    def __init__(self, handler: DatasetFamilyHandler, asset_store: AssetStore) -> None:
        self._handler = handler
        self._asset_store = asset_store

    def iter_cards(self) -> Iterable[AssetCard]:
        return self._asset_store.find_cards("dataset_family", self._handler.family)

    def get_dataset_config(self, card: AssetCard | str) -> DatasetConfigT:
        if isinstance(card, str):
            name = card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError:
                raise DatasetNotKnownError(name) from None
        else:
            name = card.name

        family = card.field("dataset_family").as_(str)

        if family != self._handler.family:
            msg = f"family field of the {name} asset card is expected to be '{self._handler.family}', but is '{family}' instead."

            raise AssetCardError(name, msg)

        config = self._handler.get_dataset_config(card)

        return cast(DatasetConfigT, config)

    def open_dataset(
        self, card: AssetCard | str, *, config: DatasetConfigT | None = None
    ) -> DatasetT:
        if isinstance(card, str):
            name = card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError:
                raise DatasetNotKnownError(name) from None
        else:
            name = card.name

        family = card.field("dataset_family").as_(str)

        if family != self._handler.family:
            msg = f"family field of the {name} asset card is expected to be {self._handler.family}, but is {family} instead."

            raise AssetCardError(name, msg)

        dataset = self._handler.open_dataset(card, config)

        return cast(DatasetT, dataset)

    def open_custom_dataset(
        self, resolver: DependencyResolver, name: str, config: DatasetConfigT
    ) -> DatasetT:
        dataset = self._handler.open_custom_dataset(name, config)

        return cast(DatasetT, dataset)

    @property
    def handler(self) -> DatasetFamilyHandler:
        return self._handler


@final
class DatasetHubAccessor(Generic[DatasetT, DatasetConfigT]):
    def __init__(
        self, family: str, kls: type[DatasetT], config_kls: type[DatasetConfigT]
    ) -> None:
        self._family = family
        self._kls = kls
        self._config_kls = config_kls

    def __call__(self) -> DatasetHub[DatasetT, DatasetConfigT]:
        resolver = get_dependency_resolver()

        asset_store = resolver.resolve(AssetStore)

        handlers = resolver.get_provider(DatasetFamilyHandler)

        family = self._family

        try:
            handler = handlers.get(family)
        except DependencyNotFoundError:
            raise DatasetFamilyNotKnownError(family) from None

        if not issubclass(handler.kls, self._kls):
            raise InternalError(
                f"`kls` is `{self._kls}`, but the type of the {family} dataset family is `{handler.kls}`."
            )

        if not issubclass(handler.config_kls, self._config_kls):
            raise InternalError(
                f"`config_kls` is `{self._config_kls}`, but the configuration type of the {family} dataset family is `{handler.config_kls}`."
            )

        return DatasetHub(handler, asset_store)


class DatasetNotKnownError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is not a known dataset.")

        self.name = name


class DatasetFamilyNotKnownError(Exception):
    def __init__(self, family: str) -> None:
        super().__init__(f"{family} is not a know dataset family.")

        self.family = family
