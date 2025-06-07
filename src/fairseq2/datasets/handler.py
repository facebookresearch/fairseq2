# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from errno import ENOENT
from os import strerror
from pathlib import Path
from typing import Protocol, final

from typing_extensions import override

from fairseq2.assets import AssetCard, AssetCardError, AssetDownloadManager
from fairseq2.datasets.error import DatasetLoadError, dataset_asset_card_error
from fairseq2.dependency import DependencyContainer, DependencyResolver
from fairseq2.file_system import FileSystem


class DatasetHandler(ABC):
    @abstractmethod
    def load(self, resolver: DependencyResolver, card: AssetCard) -> object: ...

    @abstractmethod
    def load_from_path(
        self, resolver: DependencyResolver, path: Path, name: str
    ) -> object: ...

    @property
    @abstractmethod
    def family(self) -> str: ...

    @property
    @abstractmethod
    def kls(self) -> type[object]: ...


class DatasetLoader(Protocol):
    def __call__(
        self, resolver: DependencyResolver, path: Path, name: str
    ) -> object: ...


@final
class DelegatingDatasetHandler(DatasetHandler):
    _family: str
    _kls: type[object]
    _loader: DatasetLoader
    _file_system: FileSystem
    _asset_download_manager: AssetDownloadManager

    def __init__(
        self,
        family: str,
        kls: type[object],
        loader: DatasetLoader,
        file_system: FileSystem,
        asset_download_manager: AssetDownloadManager,
    ) -> None:
        self._family = family
        self._kls = kls
        self._loader = loader
        self._file_system = file_system
        self._asset_download_manager = asset_download_manager

    @override
    def load(self, resolver: DependencyResolver, card: AssetCard) -> object:
        name = card.name

        try:
            uri = card.field("data").as_uri()
        except AssetCardError as ex:
            raise dataset_asset_card_error(name) from ex

        path = self._asset_download_manager.download_dataset(uri, name)

        try:
            return self.load_from_path(resolver, path, name)
        except FileNotFoundError:
            raise DatasetLoadError(
                name, f"The '{name}' dataset cannot be found at the '{path}' path."
            ) from None

    @override
    def load_from_path(
        self, resolver: DependencyResolver, path: Path, name: str
    ) -> object:
        try:
            path_exists = self._file_system.exists(path)
        except OSError as ex:
            raise DatasetLoadError(
                name, f"The '{path}' path of the dataset cannot be accessed. See the nested exception for details."  # fmt: skip
            ) from ex

        if not path_exists:
            raise FileNotFoundError(ENOENT, strerror(ENOENT), path)

        return self._loader(resolver, path, name)

    @property
    @override
    def family(self) -> str:
        return self._family

    @property
    @override
    def kls(self) -> type[object]:
        return self._kls


def register_dataset_family(
    container: DependencyContainer,
    family: str,
    kls: type[object],
    loader: DatasetLoader,
) -> None:
    def create_handler(resolver: DependencyResolver) -> DatasetHandler:
        file_system = resolver.resolve(FileSystem)

        asset_download_manager = resolver.resolve(AssetDownloadManager)

        return DelegatingDatasetHandler(
            family, kls, loader, file_system, asset_download_manager
        )

    container.register(DatasetHandler, create_handler, key=family)
