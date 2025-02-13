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
from fairseq2.context import get_runtime_context
from fairseq2.datasets._error import (
    InvalidDatasetTypeError,
    UnknownDatasetError,
    UnknownDatasetFamilyError,
    dataset_asset_card_error,
)
from fairseq2.datasets._handler import DatasetHandler
from fairseq2.registry import Provider

DatasetT = TypeVar("DatasetT")


@final
class DatasetHub(Generic[DatasetT]):
    _kls: type[DatasetT]
    _asset_store: AssetStore
    _dataset_handlers: Provider[DatasetHandler]

    def __init__(
        self,
        kls: type[DatasetT],
        asset_store: AssetStore,
        dataset_handlers: Provider[DatasetHandler],
    ) -> None:
        self._kls = kls
        self._asset_store = asset_store
        self._dataset_handlers = dataset_handlers

    def load(self, name_or_card: str | AssetCard) -> DatasetT:
        if isinstance(name_or_card, AssetCard):
            card = name_or_card

            dataset_name = card.name
        else:
            dataset_name = name_or_card

            try:
                card = self._asset_store.retrieve_card(dataset_name)
            except AssetCardNotFoundError:
                raise UnknownDatasetError(dataset_name) from None
            except AssetCardError as ex:
                raise dataset_asset_card_error(dataset_name) from ex

        try:
            dataset_family = card.field("dataset_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise UnknownDatasetError(dataset_name) from None
        except AssetCardError as ex:
            raise dataset_asset_card_error(dataset_name) from ex

        try:
            handler = self._dataset_handlers.get(dataset_family)
        except LookupError:
            raise UnknownDatasetFamilyError(dataset_family, dataset_name) from None

        if not issubclass(handler.kls, self._kls):
            raise InvalidDatasetTypeError(dataset_name, handler.kls, self._kls)

        dataset = handler.load(card)

        return cast(DatasetT, dataset)


@final
class DatasetHubAccessor(Generic[DatasetT]):
    _kls: type[DatasetT]

    def __init__(self, kls: type[DatasetT]) -> None:
        self._kls = kls

    def __call__(self) -> DatasetHub[DatasetT]:
        context = get_runtime_context()

        dataset_handlers = context.get_registry(DatasetHandler)

        return DatasetHub(self._kls, context.asset_store, dataset_handlers)
