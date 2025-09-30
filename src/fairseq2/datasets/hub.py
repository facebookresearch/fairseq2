# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, TypeVar, cast, final

from fairseq2.assets import AssetCard, AssetCardError, AssetNotFoundError, AssetStore
from fairseq2.datasets.family import DatasetFamily
from fairseq2.error import InternalError
from fairseq2.runtime.dependency import get_dependency_resolver

DatasetT = TypeVar("DatasetT")

DatasetConfigT = TypeVar("DatasetConfigT")


@final
class DatasetHub(Generic[DatasetT, DatasetConfigT]):
    def __init__(self, family: DatasetFamily, asset_store: AssetStore) -> None:
        self._family = family
        self._asset_store = asset_store

    def iter_cards(self) -> Iterator[AssetCard]:
        return self._asset_store.find_cards("dataset_family", self._family.name)

    def get_dataset_config(self, card: AssetCard | str) -> DatasetConfigT:
        if isinstance(card, str):
            name = card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError:
                raise DatasetNotKnownError(name) from None
        else:
            name = card.name

        family_name = card.field("dataset_family").as_(str)

        if family_name != self._family.name:
            msg = f"family field of the {name} asset card is expected to be {self._family.name}, but is {family_name} instead."

            raise AssetCardError(name, msg)

        config = self._family.get_dataset_config(card)

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

        family_name = card.field("dataset_family").as_(str)

        if family_name != self._family.name:
            msg = f"family field of the {name} asset card is expected to be {self._family.name}, but is {family_name} instead."

            raise AssetCardError(name, msg)

        dataset = self._family.open_dataset(card, config)

        return cast(DatasetT, dataset)

    def open_custom_dataset(self, config: DatasetConfigT) -> DatasetT:
        dataset = self._family.open_custom_dataset(config)

        return cast(DatasetT, dataset)


@final
class DatasetHubAccessor(Generic[DatasetT, DatasetConfigT]):
    def __init__(
        self, family_name: str, kls: type[DatasetT], config_kls: type[DatasetConfigT]
    ) -> None:
        self._family_name = family_name
        self._kls = kls
        self._config_kls = config_kls

    def __call__(self) -> DatasetHub[DatasetT, DatasetConfigT]:
        resolver = get_dependency_resolver()

        asset_store = resolver.resolve(AssetStore)

        name = self._family_name

        family = resolver.resolve_optional(DatasetFamily, key=name)
        if family is None:
            raise DatasetFamilyNotKnownError(name)

        if not issubclass(family.kls, self._kls):
            raise InternalError(
                f"`kls` is `{self._kls}`, but the type of the {name} dataset family is `{family.kls}`."
            )

        if not issubclass(family.config_kls, self._config_kls):
            raise InternalError(
                f"`config_kls` is `{self._config_kls}`, but the configuration type of the {name} dataset family is `{family.config_kls}`."
            )

        return DatasetHub(family, asset_store)


class DatasetNotKnownError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is not a known dataset.")

        self.name = name


class DatasetFamilyNotKnownError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is not a know dataset family.")

        self.name = name
