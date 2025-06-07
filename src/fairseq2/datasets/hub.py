# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Generic, TypeVar, cast, final

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetCardNotFoundError,
    AssetStore,
)
from fairseq2.datasets.error import (
    InvalidDatasetTypeError,
    UnknownDatasetError,
    UnknownDatasetFamilyError,
    dataset_asset_card_error,
)
from fairseq2.datasets.handler import DatasetHandler
from fairseq2.dependency import DependencyResolver
from fairseq2.typing import Provider

DatasetT = TypeVar("DatasetT")


@final
class DatasetHub(Generic[DatasetT]):
    _kls: type[DatasetT]
    _asset_store: AssetStore
    _handlers: Provider[DatasetHandler]
    _resolver: DependencyResolver

    def __init__(
        self,
        kls: type[DatasetT],
        asset_store: AssetStore,
        handlers: Provider[DatasetHandler],
        resolver: DependencyResolver,
    ) -> None:
        self._kls = kls
        self._asset_store = asset_store
        self._handlers = handlers
        self._resolver = resolver

    def load(self, name_or_card: str | AssetCard) -> DatasetT:
        if isinstance(name_or_card, AssetCard):
            card = name_or_card

            name = card.name
        else:
            name = name_or_card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetCardNotFoundError:
                raise UnknownDatasetError(name) from None
            except AssetCardError as ex:
                raise dataset_asset_card_error(name) from ex

        try:
            family = card.field("dataset_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise UnknownDatasetError(name) from None
        except AssetCardError as ex:
            raise dataset_asset_card_error(name) from ex

        try:
            handler = self._handlers.resolve(family)
        except LookupError:
            raise UnknownDatasetFamilyError(family, name) from None

        if not issubclass(handler.kls, self._kls):
            raise InvalidDatasetTypeError(name, handler.kls, self._kls)

        dataset = handler.load(self._resolver, card)

        return cast(DatasetT, dataset)


@final
class DatasetHubAccessor(Generic[DatasetT]):
    _kls: type[DatasetT]

    def __init__(self, kls: type[DatasetT]) -> None:
        self._kls = kls

    def __call__(self) -> DatasetHub[DatasetT]:
        from fairseq2 import get_dependency_resolver

        resolver = get_dependency_resolver()

        asset_store = resolver.resolve(AssetStore)

        handlers = resolver.resolve_provider(DatasetHandler)

        return DatasetHub(self._kls, asset_store, handlers, resolver)
