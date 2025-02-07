# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, final

from typing_extensions import override

from fairseq2.assets import AssetCard, AssetCardError, AssetDownloadManager
from fairseq2.datasets._error import dataset_asset_card_error


class DatasetHandler(ABC):
    @abstractmethod
    def load(self, card: AssetCard) -> object: ...

    @abstractmethod
    def load_from_path(self, path: Path, dataset_name: str) -> object: ...

    @property
    @abstractmethod
    def family(self) -> str: ...

    @property
    @abstractmethod
    def kls(self) -> type[object]: ...


class DatasetLoader(Protocol):
    def __call__(self, path: Path, name: str) -> object: ...


@final
class StandardDatasetHandler(DatasetHandler):
    _family: str
    _kls: type[object]
    _loader: DatasetLoader
    _asset_download_manager: AssetDownloadManager

    def __init__(
        self,
        family: str,
        kls: type[object],
        loader: DatasetLoader,
        asset_download_manager: AssetDownloadManager,
    ) -> None:
        self._family = family
        self._kls = kls
        self._loader = loader
        self._asset_download_manager = asset_download_manager

    @override
    def load(self, card: AssetCard) -> object:
        dataset_name = card.name

        try:
            dataset_uri = card.field("data").as_uri()
        except AssetCardError as ex:
            raise dataset_asset_card_error(dataset_name) from ex

        path = self._asset_download_manager.download_dataset(dataset_uri, dataset_name)

        return self._loader(path, dataset_name)

    @override
    def load_from_path(self, path: Path, dataset_name: str) -> object:
        return self._loader(path, dataset_name)

    @property
    @override
    def family(self) -> str:
        return self._family

    @property
    @override
    def kls(self) -> type[object]:
        return self._kls
