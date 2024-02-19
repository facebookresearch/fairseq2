# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Generic, Protocol, TypeVar, Union, final

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetDownloadManager,
    AssetError,
    AssetStore,
)
from fairseq2.typing import finaloverride

DatasetT = TypeVar("DatasetT", covariant=True)


class DatasetLoader(ABC, Generic[DatasetT]):
    """Loads datasets of type ``DatasetT```."""

    @abstractmethod
    def __call__(
        self,
        dataset_name_or_card: Union[str, AssetCard],
        *,
        force: bool = False,
        cache_only: bool = False,
        progress: bool = True,
    ) -> DatasetT:
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


class DatasetFactory(Protocol[DatasetT]):
    """Constructs datasets of type ``DatasetT``."""

    def __call__(self, path: Path, card: AssetCard) -> DatasetT:
        """
        :param path:
            The path to the dataset.
        :param card:
            The asset card of the dataset.
        """


class StandardDatasetLoader(DatasetLoader[DatasetT]):
    """Loads datasets of type ``DatasetT``."""

    asset_store: AssetStore
    download_manager: AssetDownloadManager

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
        self.asset_store = asset_store
        self.download_manager = download_manager
        self.factory = factory

    @finaloverride
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
            card = self.asset_store.retrieve_card(dataset_name_or_card)

        uri = card.field("uri").as_uri()

        try:
            path = self.download_manager.download_dataset(
                uri, card.name, force=force, cache_only=cache_only, progress=progress
            )
        except ValueError as ex:
            raise AssetCardError(
                f"The value of the field 'uri' of the asset card '{card.name}' is not valid. See nested exception for details."
            ) from ex

        try:
            return self.factory(path, card)
        except ValueError as ex:
            raise AssetError(
                f"The {card.name} dataset cannot be loaded. See nested exception for details."
            ) from ex


@final
class DelegatingDatasetLoader(DatasetLoader[DatasetT]):
    """Loads datasets of type ``DatasetT`` using registered loaders."""

    asset_store: AssetStore

    _loaders: Dict[str, DatasetLoader[DatasetT]]

    def __init__(self, asset_store: AssetStore) -> None:
        """
        :param asset_store:
            The asset store where to check for available datasets.
        """
        self.asset_store = asset_store

        self._loaders = {}

    @finaloverride
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
            card = self.asset_store.retrieve_card(dataset_name_or_card)

        dataset_type = card.field("dataset_type").as_(str)

        try:
            loader = self._loaders[dataset_type]
        except KeyError:
            raise RuntimeError(
                f"The dataset type '{dataset_type}' has no registered loader."
            )

        return loader(card, force=force, cache_only=cache_only, progress=progress)

    def register_loader(
        self, dataset_type: str, loader: DatasetLoader[DatasetT]
    ) -> None:
        """Register a dataset loader to use with this loader.

        :param dataset_type:
            The dataset type. If the 'dataset_type' field of an asset card
            matches this value, the specified ``loader`` will be used.
        :param loader:
            The dataset loader.
        """
        if dataset_type in self._loaders:
            raise ValueError(
                f"`dataset_type` must be a unique dataset type, but '{dataset_type}' is already registered."
            )

        self._loaders[dataset_type] = loader
