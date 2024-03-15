# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Protocol, TypeVar, Union, final

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetDownloadManager,
    AssetError,
    AssetStore,
)

DatasetT = TypeVar("DatasetT")

DatasetT_co = TypeVar("DatasetT_co", covariant=True)


class DatasetLoader(Protocol[DatasetT_co]):
    """Loads datasets of type ``DatasetT```."""

    def __call__(
        self,
        dataset_name_or_card: Union[str, AssetCard],
        *,
        force: bool = False,
        cache_only: bool = False,
        progress: bool = True,
    ) -> DatasetT_co:
        """
        :param dataset_name_or_card:
            The name or asset card of the dataset to load.
        :param force:
            If ``True``, downloads the dataset even if it is already in cache.
        :param cache_only:
            If ``True``, skips the download and uses the cached dataset.
        :param progress:
            If ``True``, displays a progress bar to stderr.
        """


class DatasetFactory(Protocol[DatasetT_co]):
    """Constructs datasets of type ``DatasetT``."""

    def __call__(self, path: Path, card: AssetCard) -> DatasetT_co:
        """
        :param path:
            The path to the dataset.
        :param card:
            The asset card of the dataset.
        """


@final
class StandardDatasetLoader(DatasetLoader[DatasetT]):
    """Loads datasets of type ``DatasetT``."""

    _asset_store: AssetStore
    _download_manager: AssetDownloadManager
    _factory: DatasetFactory[DatasetT]

    def __init__(
        self,
        asset_store: AssetStore,
        download_manager: AssetDownloadManager,
        factory: DatasetFactory[DatasetT],
    ) -> None:
        """
        :param asset_store:
            The asset store where to check for available datasets.
        :param download_manager:
            The download manager.
        :param factory:
            The factory to construct datasets.
        """
        self._asset_store = asset_store
        self._download_manager = download_manager
        self._factory = factory

    def __call__(
        self,
        dataset_name_or_card: Union[str, AssetCard],
        *,
        force: bool = False,
        cache_only: bool = False,
        progress: bool = True,
    ) -> DatasetT:
        if isinstance(dataset_name_or_card, AssetCard):
            card = dataset_name_or_card
        else:
            card = self._asset_store.retrieve_card(dataset_name_or_card)

        uri = card.field("data").as_uri()

        try:
            path = self._download_manager.download_dataset(
                uri, card.name, force=force, cache_only=cache_only, progress=progress
            )
        except ValueError as ex:
            raise AssetCardError(
                f"The value of the field 'data' of the asset card '{card.name}' is not valid. See nested exception for details."
            ) from ex

        try:
            return self._factory(path, card)
        except ValueError as ex:
            raise AssetError(
                f"The {card.name} dataset cannot be loaded. See nested exception for details."
            ) from ex


@final
class DelegatingDatasetLoader(DatasetLoader[DatasetT]):
    """Loads datasets of type ``DatasetT`` using registered loaders."""

    _asset_store: AssetStore
    _loaders: Dict[str, DatasetLoader[DatasetT]]

    def __init__(self, asset_store: AssetStore) -> None:
        """
        :param asset_store:
            The asset store where to check for available datasets.
        """
        self._asset_store = asset_store

        self._loaders = {}

    def __call__(
        self,
        dataset_name_or_card: Union[str, AssetCard],
        *,
        force: bool = False,
        cache_only: bool = False,
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
                f"The value of the field 'dataset_family' of the asset card '{card.name}' must be a supported dataset type, but '{family}' has no registered loader."
            )

        return loader(card, force=force, cache_only=cache_only, progress=progress)

    def register_loader(self, family: str, loader: DatasetLoader[DatasetT]) -> None:
        """Register a dataset loader to use with this loader.

        :param family:
            The dataset type. If the 'dataset_family' field of an asset card
            matches this value, the specified ``loader`` will be used.
        :param loader:
            The dataset loader.
        """
        if family in self._loaders:
            raise ValueError(
                f"`family` must be a unique dataset type, but '{family}' has already a registered loader."
            )

        self._loaders[family] = loader
