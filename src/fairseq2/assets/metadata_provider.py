# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, final

import yaml
from typing_extensions import NoReturn
from yaml import YAMLError

from fairseq2.assets.error import AssetError
from fairseq2.typing import finaloverride


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

    base_dir: Path

    _cache: Optional[Dict[str, Dict[str, Any]]]

    def __init__(self, base_dir: Path) -> None:
        """
        :param base_dir:
            The base directory under which asset metadata is stored.
        """
        self.base_dir = base_dir

        self._cache = None

    @finaloverride
    def get_metadata(self, name: str) -> Dict[str, Any]:
        self._ensure_cache_loaded()

        assert self._cache is not None

        try:
            return deepcopy(self._cache[name])
        except KeyError:
            raise AssetNotFoundError(
                f"An asset with the name '{name}' cannot be found."
            )

    def _ensure_cache_loaded(self) -> None:
        if self._cache is not None:
            return

        self._cache = {}

        def on_walk_error(ex: OSError) -> NoReturn:
            raise AssetMetadataError(
                f"The base asset metadata directory '{self.base_dir}' cannot be traversed. See nested exception for details."
            ) from ex

        for dir_pathname, _, filenames in os.walk(self.base_dir, onerror=on_walk_error):
            metadata_dir = Path(dir_pathname)

            for filename in filenames:
                file = metadata_dir.joinpath(filename)

                if file.suffix != ".yaml" and file.suffix != ".yml":
                    continue

                self._load_file(file)

    def _load_file(self, file: Path) -> None:
        assert self._cache is not None

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
                        f"The asset metadata at index {idx} in the file '{file}' is invalid."
                    )

                try:
                    name = metadata["name"]
                except KeyError:
                    raise AssetMetadataError(
                        f"The asset metadata at index {idx} in the file '{file}' does not have a name."
                    )

                if not isinstance(name, str):
                    raise AssetMetadataError(
                        f"The asset metadata at index {idx} in the file '{file}' has an invalid name."
                    )

                if name in self._cache:
                    raise AssetMetadataError(f"Two assets have the same name '{name}'.")

                self._cache[name] = metadata

    @finaloverride
    def clear_cache(self) -> None:
        self._cache = None


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
                    f"The asset metadata at index {idx} in `metadata` does not have a name."
                )

            if not isinstance(name, str):
                raise AssetMetadataError(
                    f"The asset metadata at index {idx} in `metadata` has an invalid name."
                )

            if name in self._metadata:
                raise AssetMetadataError(f"Two assets have the same name '{name}'.")

            self._metadata[name] = m

    @finaloverride
    def get_metadata(self, name: str) -> Dict[str, Any]:
        try:
            return deepcopy(self._metadata[name])
        except KeyError:
            raise AssetNotFoundError(
                f"An asset with the name '{name}' cannot be found."
            )

    @finaloverride
    def clear_cache(self) -> None:
        pass


class AssetNotFoundError(AssetError):
    """Raised when an asset cannot be found."""


class AssetMetadataError(AssetError):
    """Raise when an asset metadata operation fails."""
