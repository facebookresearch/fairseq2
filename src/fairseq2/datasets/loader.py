# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, TypeVar, cast, final

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetDownloadManager,
    AssetError,
    AssetStore,
    default_asset_download_manager,
    default_asset_store,
)

DatasetT = TypeVar("DatasetT")

DatasetT_co = TypeVar("DatasetT_co", covariant=True)


class DatasetLoader(Protocol[DatasetT_co]):
    """Loads datasets of type ``DatasetT```."""

    def __call__(
        self,
        dataset_name_or_card: str | AssetCard,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> DatasetT_co:
        """
        :param dataset_name_or_card:
            The name or the asset card of the dataset to load.
        :param force:
            If ``True``, downloads the dataset even if it is already in cache.
        :param progress:
            If ``True``, displays a progress bar to stderr.
        """


class AbstractDatasetLoader(ABC, DatasetLoader[DatasetT]):
    """Provides a skeletal implementation of :class:`DatasetLoader`."""

    _asset_store: AssetStore
    _download_manager: AssetDownloadManager

    def __init__(
        self,
        *,
        asset_store: AssetStore | None = None,
        download_manager: AssetDownloadManager | None = None,
    ) -> None:
        """
        :param asset_store:
            The asset store where to check for available datasets. If ``None``,
            the default asset store will be used.
        :param download_manager:
            The download manager. If ``None``, the default download manager will
            be used.
        """
        self._asset_store = asset_store or default_asset_store
        self._download_manager = download_manager or default_asset_download_manager

    @final
    def __call__(
        self,
        dataset_name_or_card: str | AssetCard,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> DatasetT:
        if isinstance(dataset_name_or_card, AssetCard):
            card = dataset_name_or_card
        else:
            card = self._asset_store.retrieve_card(dataset_name_or_card)

        dataset_uri = card.field("data").as_uri()

        try:
            path = self._download_manager.download_dataset(
                dataset_uri, card.name, force=force, progress=progress
            )
        except ValueError as ex:
            raise AssetCardError(
                f"The value of the field 'data' of the asset card '{card.name}' must be a URI. See nested exception for details."
            ) from ex

        try:
            return self._load(path, card)
        except ValueError as ex:
            raise AssetError(
                f"The {card.name} dataset cannot be loaded. See nested exception for details."
            ) from ex

    @abstractmethod
    def _load(self, path: Path, card: AssetCard) -> DatasetT:
        """
        :param path:
            The path to the dataset.
        :param card:
            The asset card of the dataset.
        """


@final
class DelegatingDatasetLoader(DatasetLoader[DatasetT]):
    """Loads datasets of type ``DatasetT`` using registered loaders."""

    _asset_store: AssetStore
    _loaders: dict[str, DatasetLoader[DatasetT]]

    def __init__(self, *, asset_store: AssetStore | None = None) -> None:
        """
        :param asset_store:
            The asset store where to check for available datasets. If ``None``,
            the default asset store will be used.
        """
        self._asset_store = asset_store or default_asset_store

        self._loaders = {}

    def __call__(
        self,
        dataset_name_or_card: str | AssetCard,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> DatasetT:
        if isinstance(dataset_name_or_card, AssetCard):
            card = dataset_name_or_card
        else:
            card = self._asset_store.retrieve_card(dataset_name_or_card)

        family = card.field("dataset_family").as_(str)

        try:
            loader = self._loaders[family]
        except KeyError:
            raise AssetError(
                f"The value of the field 'dataset_family' of the asset card '{card.name}' must be a supported dataset family, but is '{family}' instead."
            ) from None

        return loader(card, force=force, progress=progress)

    def register(self, family: str, loader: DatasetLoader[DatasetT]) -> None:
        """Register a dataset loader to use with this loader.

        :param family:
            The dataset type. If the 'dataset_family' field of an asset card
            matches this value, the specified ``loader`` will be used.
        :param loader:
            The dataset loader.
        """
        if family in self._loaders:
            raise ValueError(
                f"`family` must be a unique dataset family name, but '{family}' is already registered."
            )

        self._loaders[family] = loader

    def supports(self, dataset_name_or_card: str | AssetCard) -> bool:
        """Return ``True`` if the specified dataset has a registered loader."""
        if isinstance(dataset_name_or_card, AssetCard):
            card = dataset_name_or_card
        else:
            card = self._asset_store.retrieve_card(dataset_name_or_card)

        family = card.field("dataset_family").as_(str)

        return family in self._loaders


def is_dataset_card(card: AssetCard) -> bool:
    """Return ``True`` if ``card`` specifies a dataset."""
    return card.field("dataset_family").exists()


def get_dataset_family(card: AssetCard) -> str:
    """Return the dataset family name contained in ``card``."""
    return cast(str, card.field("dataset_family").as_(str))
