# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, TypeVar, cast, final

from fairseq2.assets import (
    AssetCard,
    AssetCardFieldNotFoundError,
    AssetNotFoundError,
    AssetStore,
)
from fairseq2.datasets.error import (
    DatasetError,
    UnknownDatasetError,
    UnknownDatasetFamilyError,
)
from fairseq2.datasets.handler import DatasetFamilyHandler
from fairseq2.error import ContractError
from fairseq2.runtime.dependency import (
    DependencyNotFoundError,
    DependencyResolver,
    get_dependency_resolver,
)

DatasetT = TypeVar("DatasetT")


@final
class DatasetHub(Generic[DatasetT]):
    _handler: DatasetFamilyHandler
    _asset_store: AssetStore
    _resolver: DependencyResolver

    def __init__(
        self,
        handler: DatasetFamilyHandler,
        asset_store: AssetStore,
        resolver: DependencyResolver,
    ) -> None:
        self._handler = handler
        self._asset_store = asset_store
        self._resolver = resolver

    def iter_dataset_cards(self) -> Iterable[AssetCard]:
        return self._asset_store.find_cards("dataset_family", self._handler.family)

    def open_dataset(self, name_or_card: str | AssetCard) -> DatasetT:
        if isinstance(name_or_card, AssetCard):
            card = name_or_card

            name = card.name
        else:
            name = name_or_card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError:
                raise UnknownDatasetError(name) from None

        try:
            family = card.field("dataset_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise UnknownDatasetError(name) from None

        if family != self._handler.family:
            raise DatasetError(
                name, f"The '{name}' dataset is not of family '{family}'."
            )

        dataset = self._handler.open_dataset(self._resolver, card)

        return cast(DatasetT, dataset)


@final
class DatasetHubAccessor(Generic[DatasetT]):
    _family: str
    _kls: type[DatasetT]

    def __init__(self, family: str, kls: type[DatasetT]) -> None:
        self._family = family
        self._kls = kls

    def __call__(self) -> DatasetHub[DatasetT]:
        resolver = get_dependency_resolver()

        asset_store = resolver.resolve(AssetStore)

        handlers = resolver.get_provider(DatasetFamilyHandler)

        family = self._family

        try:
            handler = handlers.get(family)
        except DependencyNotFoundError:
            raise UnknownDatasetFamilyError(family) from None

        if not issubclass(handler.kls, self._kls):
            raise ContractError(
                f"`kls` is `{self._kls}`, but the type of the '{family}' dataset family is `{handler.kls}`."
            )

        return DatasetHub(handler, asset_store, resolver)
