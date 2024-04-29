# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, final

import yaml
from importlib_resources import files
from importlib_resources.readers import MultiplexedPath
from typing_extensions import NoReturn
from yaml import YAMLError

from fairseq2.assets.error import AssetError
from fairseq2.typing import override


class AssetMetadataProvider(ABC):
    """Provides asset metadata."""

    @abstractmethod
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Return the metadata of the specified asset.

        :param name:
            The name of the asset.
        """

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear any cached asset metadata."""


@final
class FileAssetMetadataProvider(AssetMetadataProvider):
    """Provides asset metadata stored on a file system."""

    _base_dir: Path
    _cache: Optional[Dict[str, Dict[str, Any]]]

    def __init__(self, base_dir: Path) -> None:
        """
        :param base_dir:
            The base directory under which asset metadata is stored.
        """
        self._base_dir = base_dir

        self._cache = None

    @override
    def get_metadata(self, name: str) -> Dict[str, Any]:
        self._ensure_cache_loaded()

        try:
            return deepcopy(self._cache[name])  # type: ignore[index]
        except KeyError:
            raise AssetNotFoundError(
                f"An asset with the name '{name}' cannot be found."
            )

    def _ensure_cache_loaded(self) -> None:
        if self._cache is not None:
            return

        self._cache = {}

        def on_error(ex: OSError) -> NoReturn:
            raise AssetMetadataError(
                f"The base asset metadata directory '{self._base_dir}' cannot be traversed. See nested exception for details."
            ) from ex

        for dir_pathname, _, filenames in os.walk(self._base_dir, onerror=on_error):
            metadata_dir = Path(dir_pathname)

            for filename in filenames:
                file = metadata_dir.joinpath(filename)

                if file.suffix != ".yaml" and file.suffix != ".yml":
                    continue

                for name, metadata in _load_metadata_file(file):
                    if name in self._cache:
                        raise AssetMetadataError(
                            f"Two assets under the directory '{self._base_dir}' have the same name '{name}'."
                        )

                    self._cache[name] = metadata

    @override
    def clear_cache(self) -> None:
        self._cache = None


@final
class PackageAssetMetadataProvider(AssetMetadataProvider):
    """Provides asset metadata stored in a Python namespace package."""

    _package_name: str
    _package_path: MultiplexedPath
    _cache: Optional[Dict[str, Dict[str, Any]]]

    def __init__(self, package_name: str) -> None:
        """
        :param package_name:
            The name of the package in which asset metadata is stored.
        """
        self._package_name = package_name

        self._package_path = files(package_name)

        self._cache = None

    @override
    def get_metadata(self, name: str) -> Dict[str, Any]:
        self._ensure_cache_loaded()

        try:
            return deepcopy(self._cache[name])  # type: ignore[index]
        except KeyError:
            raise AssetNotFoundError(
                f"An asset with the name '{name}' cannot be found."
            )

    def _ensure_cache_loaded(self) -> None:
        if self._cache is not None:
            return

        self._cache = {}

        for file in self._list_files():
            if file.suffix != ".yaml" and file.suffix != ".yml":
                continue

            for name, metadata in _load_metadata_file(file):
                if name in self._cache:
                    raise AssetMetadataError(
                        f"Two assets under the namespace package '{self._package_name}' have the same name '{name}'."
                    )

                self._cache[name] = metadata

    def _list_files(self) -> List[Path]:
        files = []

        def collect_files(p: Union[MultiplexedPath, Path]) -> None:
            if p.is_file():
                if not isinstance(p, Path):
                    raise RuntimeError(
                        "`importlib.resources` returned a file path that is not of type `pathlib.Path`. Please file a bug report."
                    )

                files.append(p)
            elif p.is_dir():
                for e in p.iterdir():
                    collect_files(e)

        collect_files(self._package_path)

        return files

    @override
    def clear_cache(self) -> None:
        self._cache = None


def _load_metadata_file(file: Path) -> List[Tuple[str, Dict[str, Any]]]:
    output = []

    try:
        fp = file.open()
    except OSError as ex:
        raise AssetMetadataError(
            f"The asset metadata file '{file}' cannot be opened. See nested exception for details."
        ) from ex

    with fp:
        try:
            all_metadata = yaml.safe_load_all(fp)
        except YAMLError as ex:
            raise AssetMetadataError(
                f"The asset metadata file '{file}' cannot be loaded. See nested exception for details."
            ) from ex

        for idx, metadata in enumerate(all_metadata):
            if not isinstance(metadata, dict):
                raise AssetMetadataError(
                    f"The asset metadata at index {idx} in the file '{file}' has an invalid format."
                )

            try:
                name = metadata["name"]
            except KeyError:
                raise AssetMetadataError(
                    f"The asset metadata at index {idx} in the file '{file}' does not have a name entry."
                )

            if not isinstance(name, str):
                raise AssetMetadataError(
                    f"The asset metadata at index {idx} in the file '{file}' has an invalid name."
                )

            output.append((name, metadata))

    return output


@final
class InProcAssetMetadataProvider(AssetMetadataProvider):
    """Provides asset metadata stored in memory."""

    _metadata: Dict[str, Dict[str, Any]]

    def __init__(self, metadata: Sequence[Dict[str, Any]]) -> None:
        self._metadata = {}

        for idx, m in enumerate(metadata):
            try:
                name = m["name"]
            except KeyError:
                raise AssetMetadataError(
                    f"The asset metadata at index {idx} in `metadata` does not have a name entry."
                )

            if not isinstance(name, str):
                raise AssetMetadataError(
                    f"The asset metadata at index {idx} in `metadata` has an invalid name."
                )

            if name in self._metadata:
                raise AssetMetadataError(
                    f"Two assets in `metadata` have the same name '{name}'."
                )

            self._metadata[name] = m

    @override
    def get_metadata(self, name: str) -> Dict[str, Any]:
        try:
            return deepcopy(self._metadata[name])
        except KeyError:
            raise AssetNotFoundError(
                f"An asset with the name '{name}' cannot be found."
            )

    @override
    def clear_cache(self) -> None:
        pass


class AssetNotFoundError(AssetError):
    """Raised when an asset cannot be found."""


class AssetMetadataError(AssetError):
    """Raise when an asset metadata operation fails."""
