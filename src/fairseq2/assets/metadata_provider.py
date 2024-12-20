# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import NoReturn, final

from importlib_resources import files as get_files
from importlib_resources.readers import MultiplexedPath
from typing_extensions import override

from fairseq2.error import ContractError, InternalError
from fairseq2.utils.file import FileSystem
from fairseq2.utils.yaml import YamlError, YamlLoader


class AssetMetadataProvider(ABC):
    """Provides asset metadata."""

    @abstractmethod
    def get_metadata(self, name: str) -> dict[str, object]:
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


class AbstractAssetMetadataProvider(AssetMetadataProvider):
    """Provides a skeletal implementation of :class:`AssetMetadataProvider`."""

    _cache: dict[str, dict[str, object]] | None

    def __init__(self) -> None:
        """
        :param scope:
            The scope of the provider.
        """
        self._cache = None

    @final
    @override
    def get_metadata(self, name: str) -> dict[str, object]:
        cache = self._ensure_cache_loaded()

        try:
            metadata = cache[name]
        except KeyError:
            raise AssetMetadataNotFoundError(
                f"An asset metadata with name '{name}' is not found."
            ) from None

        try:
            return deepcopy(metadata)
        except Exception as ex:
            raise ContractError(
                f"The metadata of the '{name}' asset cannot be copied. See the nested exception for details."
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

    def _ensure_cache_loaded(self) -> dict[str, dict[str, object]]:
        if self._cache is not None:
            return self._cache

        self._cache = self._load_cache()

        return self._cache

    @abstractmethod
    def _load_cache(self) -> dict[str, dict[str, object]]:
        ...


@final
class FileAssetMetadataProvider(AbstractAssetMetadataProvider):
    """Provides asset metadata stored on a file system."""

    _path: Path
    _file_system: FileSystem
    _yaml_loader: YamlLoader

    def __init__(
        self, path: Path, file_system: FileSystem, yaml_loader: YamlLoader
    ) -> None:
        super().__init__()

        self._path = path
        self._file_system = file_system
        self._yaml_loader = yaml_loader

    @override
    def _load_cache(self) -> dict[str, dict[str, object]]:
        path = self._file_system.resolve(self._path)

        cache = {}

        def cache_file(file: Path, source: str) -> None:
            for name, metadata in load_metadata_file(file, self._yaml_loader):
                if name in cache:
                    if file == path:
                        raise AssetMetadataError(
                            f"Two assets in the '{path}' file have the same name '{name}'."
                        )
                    else:
                        raise AssetMetadataError(
                            f"Two assets under the '{path}' directory have the same name '{name}'."
                        )

                metadata["__source__"] = source

                cache[name] = metadata

        if path.is_dir():
            source = f"directory:{path}"

            def on_error(ex: OSError) -> NoReturn:
                raise AssetMetadataError(
                    f"The '{path}' base asset metadata directory cannot be traversed. See the nested exception for details."
                ) from ex

            for dir_pathname, filenames in self._file_system.walk_directory(
                path, on_error=on_error
            ):
                metadata_dir = Path(dir_pathname)

                for filename in filenames:
                    file = metadata_dir.joinpath(filename)

                    if file.suffix != ".yaml" and file.suffix != ".yml":
                        continue

                    cache_file(file, source)
        else:
            cache_file(path, source=f"file:{path}")

        return cache


@final
class PackageAssetMetadataProvider(AbstractAssetMetadataProvider):
    """Provides asset metadata stored in a Python namespace package."""

    _package_name: str
    _package_file_lister: PackageFileLister
    _yaml_loader: YamlLoader

    def __init__(
        self,
        package_name: str,
        package_file_lister: PackageFileLister,
        yaml_loader: YamlLoader,
    ) -> None:
        super().__init__()

        self._package_name = package_name
        self._package_file_lister = package_file_lister
        self._yaml_loader = yaml_loader

    @override
    def _load_cache(self) -> dict[str, dict[str, object]]:
        source = f"package:{self._package_name}"

        cache = {}

        for file in self._package_file_lister.list(self._package_name):
            if file.suffix != ".yaml" and file.suffix != ".yml":
                continue

            for name, metadata in load_metadata_file(file, self._yaml_loader):
                if name in cache:
                    raise AssetMetadataError(
                        f"Two assets in the '{self._package_name}' package have the same name '{name}'."
                    )

                metadata["__source__"] = source

                cache[name] = metadata

        return cache


class PackageFileLister(ABC):
    @abstractmethod
    def list(self, package_name: str) -> list[Path]:
        ...


@final
class WheelPackageFileLister(PackageFileLister):
    @override
    def list(self, package_name: str) -> list[Path]:
        files = []

        def collect_files(p: MultiplexedPath | Path) -> None:
            if p.is_file():
                if not isinstance(p, Path):
                    raise InternalError(
                        f"`importlib.resources` returned a path of type `{type(p)}`."
                    )

                files.append(p)
            elif p.is_dir():
                for e in p.iterdir():
                    collect_files(e)

        path = get_files(package_name)

        collect_files(path)

        return files


def load_metadata_file(
    file: Path, yaml_loader: YamlLoader
) -> list[tuple[str, dict[str, object]]]:
    """Load asset metadata included in ``file``."""
    output = []

    try:
        all_metadata = yaml_loader(file)
    except (OSError, YamlError) as ex:
        raise AssetMetadataError(
            f"The '{file}' asset metadata file cannot be loaded as YAML. See the nested exception for details."
        ) from ex

    for idx, metadata in enumerate(all_metadata):
        if not isinstance(metadata, dict):
            raise AssetMetadataError(
                f"The asset metadata at index {idx} in the '{file}' file is expected to be of type `dict`, but is of type `{type(metadata)}` instead."
            )

        try:
            name = metadata.pop("name")
        except KeyError:
            raise AssetMetadataError(
                f"The asset metadata at index {idx} in the '{file}' file does not have a name."
            ) from None

        try:
            canonical_name = _canonicalize_name(name)
        except ValueError as ex:
            raise AssetMetadataError(
                f"The asset metadata at index {idx} in the '{file}' file does not have a valid name. See the nested exception for details."
            ) from ex

        base = metadata.get("base")
        if base is not None:
            if not isinstance(base, str) or "@" in base:
                raise AssetMetadataError(
                    f"The asset metadata at index {idx} in the '{file}' file does not have a valid base name."
                )

        metadata["__base_path__"] = file.parent

        output.append((canonical_name, metadata))

    return output


@final
class InProcAssetMetadataProvider(AssetMetadataProvider):
    """Provides asset metadata stored in memory."""

    _metadata: dict[str, dict[str, object]]
    _scope: str

    def __init__(self, metadata: Sequence[dict[str, object]]) -> None:
        super().__init__()

        self._metadata = {}

        for idx, metadata_ in enumerate(metadata):
            try:
                name = metadata_.pop("name")
            except KeyError:
                raise AssetMetadataError(
                    f"The asset metadata at index {idx} in `metadata` does not have a name."
                ) from None

            try:
                canonical_name = _canonicalize_name(name)
            except ValueError as ex:
                raise AssetMetadataError(
                    f"The asset metadata at index {idx} in `metadata` does not have a valid name. See the nested exception for details."
                ) from ex

            if canonical_name in self._metadata:
                raise AssetMetadataError(
                    f"Two assets in `metadata` have the same name '{canonical_name}'."
                )

            base = metadata_.get("base")
            if base is not None:
                if not isinstance(base, str) or "@" in base:
                    raise AssetMetadataError(
                        f"The asset metadata at index {idx} in `metadata` file does not have a valid base name."
                    )

            metadata_["__source__"] = "inproc"

            self._metadata[canonical_name] = metadata_

    @override
    def get_metadata(self, name: str) -> dict[str, object]:
        try:
            return deepcopy(self._metadata[name])
        except KeyError:
            raise AssetMetadataNotFoundError(
                f"An asset metadata with name '{name}' is not found."
            ) from None

    @override
    def get_names(self) -> list[str]:
        return list(self._metadata.keys())

    @override
    def clear_cache(self) -> None:
        pass


class AssetMetadataError(Exception):
    pass


class AssetMetadataNotFoundError(AssetMetadataError):
    pass


def _canonicalize_name(name: object) -> str:
    if not isinstance(name, str):
        raise ValueError(
            f"`name` must be of type `str`, but is of type `{type(name)}` instead."
        )

    name_env_pair = name.split("@")

    if len(name_env_pair) > 2:
        raise ValueError(
            "'@' is a reserved character and must not occur more than once in `name`."
        )

    if len(name_env_pair) == 1:
        name_env_pair.append("")  # empty env

    return "@".join(name_env_pair)
