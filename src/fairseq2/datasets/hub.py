# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Generic, TypeVar, cast, final

from fairseq2.assets import AssetCard, AssetStore
from fairseq2.context import get_runtime_context
from fairseq2.datasets.handler import (
    DatasetHandler,
    DatasetNotFoundError,
    get_dataset_family,
)
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
        else:
            card = self._asset_store.retrieve_card(name_or_card)

        family = get_dataset_family(card)

        try:
            handler = self._dataset_handlers.get(family)
        except LookupError:
            raise DatasetNotFoundError(card.name) from None

        if not issubclass(handler.kls, self._kls):
            raise TypeError(
                f"The '{card.name}' dataset is expected to be of type `{self._kls}`, but is of type `{handler.kls}` instead."
            )

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
