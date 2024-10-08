# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, NoReturn, final

import yaml
from importlib_resources import files
from importlib_resources.readers import MultiplexedPath
from typing_extensions import override
from yaml import YAMLError

from fairseq2.assets.error import AssetError
from fairseq2.dependency import DependencyContainer, DependencyResolver
from fairseq2.logging import get_log_writer
from fairseq2.utils.env import get_path_from_env

log = get_log_writer(__name__)


class AssetMetadataProvider(ABC):
    """Provides asset metadata."""

    @abstractmethod
    def get_metadata(self, name: str) -> dict[str, Any]:
        """Return the metadata of the specified asset.

        :param name:
            The name of the asset.
        """

    @abstractmethod
    def get_names(self) -> list[str]:
        """Return the names of the assets for which this provider has metadata."""

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear any cached asset metadata."""

    @property
    @abstractmethod
    def scope(self) -> str:
        """The scope (e.g. user or global) of this provider."""


class AbstractAssetMetadataProvider(AssetMetadataProvider):
    """Provides a skeletal implementation of :class:`AssetMetadataProvider`."""

    _cache: dict[str, dict[str, Any]] | None
    _scope: str

    def __init__(self, *, scope: Literal["global", "user"] = "global") -> None:
        """
        :param scope:
            The scope of the provider.
        """
        self._cache = None

        self._scope = scope

    @final
    @override
    def get_metadata(self, name: str) -> dict[str, Any]:
        cache = self._ensure_cache_loaded()

        try:
            metadata = cache[name]
        except KeyError:
            raise AssetNotFoundError(
                name, f"An asset with the name '{name}' cannot be found."
            ) from None

        try:
            return deepcopy(metadata)
        except Exception as ex:
            raise AssetMetadataError(
                f"The metadata of the asset '{name}' cannot be used. Please file a bug report to the asset owner."
            ) from ex

    @final
    @override
    def get_names(self) -> list[str]:
        cache = self._ensure_cache_loaded()

        return list(cache.keys())

    @final
    @override
    def clear_cache(self) -> None:
        self._cache = None

    def _ensure_cache_loaded(self) -> dict[str, dict[str, Any]]:
        if self._cache is not None:
            return self._cache

        self._cache = self._load_cache()

        return self._cache

    @abstractmethod
    def _load_cache(self) -> dict[str, dict[str, Any]]:
        ...

    @final
    @property
    @override
    def scope(self) -> str:
        return self._scope


@final
class FileAssetMetadataProvider(AbstractAssetMetadataProvider):
    """Provides asset metadata stored on a file system."""

    _base_dir: Path

    def __init__(
        self, base_dir: Path, *, scope: Literal["global", "user"] = "global"
    ) -> None:
        """
        :param base_dir:
            The base directory under which the asset metadata is stored.
        :param scope:
            The scope of the provider.
        """
        super().__init__(scope=scope)

        self._base_dir = base_dir.expanduser().resolve()

        self._cache = None

    @override
    def _load_cache(self) -> dict[str, dict[str, Any]]:
        def on_error(ex: OSError) -> NoReturn:
            raise AssetMetadataError(
                f"The base asset metadata directory '{self._base_dir}' cannot be traversed. See nested exception for details."
            ) from ex

        cache = {}

        for dir_pathname, _, filenames in os.walk(self._base_dir, onerror=on_error):
            metadata_dir = Path(dir_pathname)

            for filename in filenames:
                file = metadata_dir.joinpath(filename)

                if file.suffix != ".yaml" and file.suffix != ".yml":
                    continue

                for name, metadata in load_metadata_file(file):
                    if name in cache:
                        raise AssetMetadataError(
                            f"Two assets under the directory '{self._base_dir}' have the same name '{name}'."
                        )

                    metadata["__source__"] = f"directory:{self._base_dir}"

                    cache[name] = metadata

        return cache


@final
class PackageAssetMetadataProvider(AbstractAssetMetadataProvider):
    """Provides asset metadata stored in a Python namespace package."""

    _package_name: str
    _package_path: MultiplexedPath

    def __init__(
        self, package_name: str, scope: Literal["global", "user"] = "global"
    ) -> None:
        """
        :param package_name:
            The name of the package in which the asset metadata is stored.
        :param scope:
            The scope of the provider.
        """
        super().__init__(scope=scope)

        self._package_name = package_name

        self._package_path = files(package_name)

    @override
    def _load_cache(self) -> dict[str, dict[str, Any]]:
        cache = {}

        for file in self._list_files():
            if file.suffix != ".yaml" and file.suffix != ".yml":
                continue

            for name, metadata in load_metadata_file(file):
                if name in cache:
                    raise AssetMetadataError(
                        f"Two assets under the package '{self._package_name}' have the same name '{name}'."
                    )

                metadata["__source__"] = f"package:{self._package_name}"

                cache[name] = metadata

        return cache

    def _list_files(self) -> list[Path]:
        files = []

        def collect_files(p: MultiplexedPath | Path) -> None:
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


def load_metadata_file(file: Path) -> list[tuple[str, dict[str, Any]]]:
    """Load asset metadata included in ``file``."""
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
        except (OSError, YAMLError) as ex:
            raise AssetMetadataError(
                f"The asset metadata file '{file}' cannot be loaded. See nested exception for details."
            ) from ex

        for idx, metadata in enumerate(all_metadata):
            if not isinstance(metadata, dict):
                raise AssetMetadataError(
                    f"The asset metadata at index {idx} in the file '{file}' has an invalid format."
                )

            try:
                name = metadata.pop("name")
            except KeyError:
                raise AssetMetadataError(
                    f"The asset metadata at index {idx} in the file {file} does not have a name entry."
                ) from None

            try:
                canonical_name = _canonicalize_name(name)
            except ValueError as ex:
                raise AssetMetadataError(
                    f"The asset metadata at index {idx} in the file {file} has an invalid name. See nested exception for details."
                ) from ex

            metadata["__base_path__"] = file.parent

            output.append((canonical_name, metadata))

    return output


@final
class InProcAssetMetadataProvider(AssetMetadataProvider):
    """Provides asset metadata stored in memory."""

    _metadata: dict[str, dict[str, Any]]
    _scope: str

    def __init__(
        self,
        metadata: Sequence[dict[str, Any]],
        *,
        scope: Literal["global", "user"] = "global",
    ) -> None:
        super().__init__()

        self._metadata = {}

        for idx, metadata_ in enumerate(metadata):
            try:
                name_ = metadata_.pop("name")
            except KeyError:
                raise AssetMetadataError(
                    f"The asset metadata at index {idx} in `metadata` does not have a name entry."
                ) from None

            try:
                canonical_name = _canonicalize_name(name_)
            except ValueError as ex:
                raise AssetMetadataError(
                    f"The asset metadata at index {idx} in `metadata` has an invalid name. See nested exception for details."
                ) from ex

            if canonical_name in self._metadata:
                raise AssetMetadataError(
                    f"Two assets in `metadata` have the same name '{canonical_name}'."
                )

            metadata_["__source__"] = "inproc"

            self._metadata[canonical_name] = metadata_

        self._scope = scope

    @override
    def get_metadata(self, name: str) -> dict[str, Any]:
        try:
            return deepcopy(self._metadata[name])
        except KeyError:
            raise AssetNotFoundError(
                name, f"An asset with the name '{name}' cannot be found."
            ) from None

    @override
    def get_names(self) -> list[str]:
        return list(self._metadata.keys())

    @override
    def clear_cache(self) -> None:
        pass

    @override
    @property
    def scope(self) -> str:
        return self._scope


def _canonicalize_name(name: Any) -> str:
    if not isinstance(name, str):
        raise ValueError(
            f"`name` must be of type `{str}`, but is of type `{type(name)}` instead."
        )

    name_env_pair = name.split("@")

    if len(name_env_pair) > 2:
        raise ValueError(
            "'@' is a reserved character and must not occur more than once in `name`."
        )

    if len(name_env_pair) == 1:
        name_env_pair.append("")  # empty env

    return "@".join(name_env_pair)


class AssetNotFoundError(AssetError):
    """Raised when an asset cannot be found."""

    _name: str

    def __init__(self, name: str, msg: str) -> None:
        super().__init__(msg)

        self._name = name

    @property
    def name(self) -> str:
        """The name of the asset."""
        return self._name


class AssetMetadataError(AssetError):
    """Raised when an asset metadata operation fails."""


def register_asset_metadata_providers(container: DependencyContainer) -> None:
    container.register_factory(AssetMetadataProvider, _create_package_metadata_provider)
    container.register_factory(AssetMetadataProvider, _create_etc_dir_metadata_provider)
    container.register_factory(AssetMetadataProvider, _create_cfg_dir_metadata_provider)


def _create_package_metadata_provider(
    resolver: DependencyResolver,
) -> AssetMetadataProvider:
    return PackageAssetMetadataProvider("fairseq2.assets.cards")


def _create_etc_dir_metadata_provider(
    resolver: DependencyResolver,
) -> AssetMetadataProvider | None:
    asset_dir = get_path_from_env("FAIRSEQ2_ASSET_DIR", log)
    if asset_dir is None:
        asset_dir = Path("/etc/fairseq2/assets").resolve()
        if not asset_dir.exists():
            return None

    return FileAssetMetadataProvider(asset_dir)


def _create_cfg_dir_metadata_provider(
    resolver: DependencyResolver,
) -> AssetMetadataProvider | None:
    asset_dir = get_path_from_env("FAIRSEQ2_USER_ASSET_DIR", log)
    if asset_dir is None:
        asset_dir = get_path_from_env("XDG_CONFIG_HOME", log)
        if asset_dir is None:
            asset_dir = Path("~/.config").expanduser()

        asset_dir = asset_dir.joinpath("fairseq2/assets").resolve()
        if not asset_dir.exists():
            return None

    return FileAssetMetadataProvider(asset_dir, scope="user")


def register_package_metadata_provider(
    container: DependencyContainer, package_name: str
) -> None:
    def create(resolver: DependencyResolver) -> AssetMetadataProvider:
        return PackageAssetMetadataProvider(package_name)

    container.register_factory(AssetMetadataProvider, create)
