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

from fairseq2.assets import AssetCard, AssetDownloadManager
from fairseq2.datasets.error import DatasetError
from fairseq2.error import InfraError
from fairseq2.file_system import FileSystem
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


class DatasetFamilyHandler(ABC):
    @abstractmethod
    def open_dataset(self, resolver: DependencyResolver, card: AssetCard) -> object: ...

    @abstractmethod
    def open_dataset_from_path(
        self, resolver: DependencyResolver, path: Path
    ) -> object: ...

    @property
    @abstractmethod
    def family(self) -> str: ...

    @property
    @abstractmethod
    def kls(self) -> type[object]: ...


class DatasetOpener(Protocol):
    def __call__(
        self, resolver: DependencyResolver, path: Path, name: str
    ) -> object: ...


@final
class StandardDatasetFamilyHandler(DatasetFamilyHandler):
    _family: str
    _kls: type[object]
    _opener: DatasetOpener
    _file_system: FileSystem
    _asset_download_manager: AssetDownloadManager

    def __init__(
        self,
        family: str,
        kls: type[object],
        opener: DatasetOpener,
        file_system: FileSystem,
        asset_download_manager: AssetDownloadManager,
    ) -> None:
        self._family = family
        self._kls = kls
        self._opener = opener
        self._file_system = file_system
        self._asset_download_manager = asset_download_manager

    @override
    def open_dataset(self, resolver: DependencyResolver, card: AssetCard) -> object:
        name = card.name

        uri = card.field("data").as_uri()

        path = self._asset_download_manager.download_dataset(uri, name)

        try:
            return self._do_open_dataset(resolver, path, name)
        except FileNotFoundError as ex:
            raise DatasetError(
                name, f"The '{name}' dataset cannot be found at the '{path}' path."
            ) from ex

    @override
    def open_dataset_from_path(
        self, resolver: DependencyResolver, path: Path
    ) -> object:
        name = str(path)

        return self._do_open_dataset(resolver, path, name)

    def _do_open_dataset(
        self, resolver: DependencyResolver, path: Path, name: str
    ) -> object:
        try:
            path_exists = self._file_system.exists(path)
        except OSError as ex:
            raise InfraError(
                f"A system error has occurred while accessing the '{path}' path of the dataset. See the nested exception for details."
            ) from ex

        if not path_exists:
            raise FileNotFoundError(ENOENT, strerror(ENOENT), path)

        return self._opener(resolver, path, name)

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
    opener: DatasetOpener,
) -> None:
    def create_handler(resolver: DependencyResolver) -> DatasetFamilyHandler:
        file_system = resolver.resolve(FileSystem)

        asset_download_manager = resolver.resolve(AssetDownloadManager)

        return StandardDatasetFamilyHandler(
            family, kls, opener, file_system, asset_download_manager
        )

    container.register(DatasetFamilyHandler, create_handler, key=family)
