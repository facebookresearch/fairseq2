# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence, Set
from copy import deepcopy
from pathlib import Path
from pickle import PickleError
from typing import NoReturn, final

from importlib_resources import files as get_files
from importlib_resources.readers import MultiplexedPath
from typing_extensions import override

from fairseq2.assets.dirs import AssetDirectoryAccessor
from fairseq2.error import OperationalError, raise_operational_system_error
from fairseq2.file_system import FileSystem
from fairseq2.utils.yaml import YamlError, YamlLoader


class AssetMetadataProvider(ABC):
    """Provides asset metadata."""

    @abstractmethod
    def maybe_get_metadata(self, name: str) -> dict[str, object] | None:
        """
        Returns the metadata of the specified asset.

        :param name: The name of the asset.
        """

    @property
    @abstractmethod
    def asset_names(self) -> Set[str]:
        """Gets the names of the assets for which this provider has metadata."""

    @property
    @abstractmethod
    def source(self) -> str: ...


class AssetMetadataError(Exception):
    def __init__(self, source: str, message: str) -> None:
        super().__init__(message)

        self.source = source


@final
class CachedAssetMetadataProvider(AssetMetadataProvider):
    def __init__(self, source: str, metadata: dict[str, dict[str, object]]) -> None:
        self._source = source
        self._metadata = metadata

    @override
    def maybe_get_metadata(self, name: str) -> dict[str, object] | None:
        asset_metadata = self._metadata.get(name)
        if asset_metadata is None:
            return None

        try:
            return deepcopy(asset_metadata)
        except (RuntimeError, TypeError, ValueError, PickleError) as ex:
            msg = f"Metadata of the {name} asset cannot be copied."

            raise AssetMetadataError(self._source, msg) from ex

    @property
    @override
    def asset_names(self) -> Set[str]:
        return self._metadata.keys()

    @property
    @override
    def source(self) -> str:
        return self._source


class AssetMetadataSource(ABC):
    @abstractmethod
    def load(self) -> Iterator[AssetMetadataProvider]: ...


class AssetSourceNotFoundError(Exception):
    def __init__(self, source: str) -> None:
        super().__init__(f"{source} asset source is not found.")

        self.source = source


class FileAssetMetadataLoader(ABC):
    @abstractmethod
    def load(self, path: Path) -> AssetMetadataProvider: ...


@final
class StandardFileAssetMetadataLoader(FileAssetMetadataLoader):
    """Provides asset metadata stored on a file system."""

    def __init__(
        self,
        file_system: FileSystem,
        metadata_file_loader: AssetMetadataFileLoader,
    ) -> None:
        self._file_system = file_system
        self._metadata_file_loader = metadata_file_loader

    @override
    def load(self, path: Path) -> AssetMetadataProvider:
        source = f"path:{path}"

        path = self._file_system.resolve(path)

        metadata = {}

        def cache_file(file: Path, source: str) -> None:
            file_metadata = self._metadata_file_loader.load(file, source)

            for name, asset_metadata in file_metadata:
                if name in metadata:
                    msg = f"Two assets at {path} have the same name {name}."

                    raise AssetMetadataError(source, msg)

                metadata[name] = asset_metadata

        is_dir = self._file_system.is_dir(path)
        if is_dir:

            def on_error(ex: OSError) -> NoReturn:
                raise ex

            for dir_pathname, filenames in self._file_system.walk_directory(
                path, on_error=on_error
            ):
                metadata_dir = Path(dir_pathname)

                for filename in filenames:
                    file = metadata_dir.joinpath(filename)

                    if file.suffix != ".yaml" and file.suffix != ".yml":
                        continue

                    try:
                        cache_file(file, source)
                    except FileNotFoundError:
                        pass
        else:
            try:
                cache_file(path, source)
            except FileNotFoundError:
                raise AssetSourceNotFoundError(source)

        return CachedAssetMetadataProvider(source, metadata)


class PackageAssetMetadataLoader(ABC):
    @abstractmethod
    def load(self, package: str) -> AssetMetadataProvider: ...


@final
class StandardPackageAssetMetadataLoader(PackageAssetMetadataLoader):
    """Provides asset metadata stored in a Python namespace package."""

    def __init__(
        self,
        file_lister: PackageFileLister,
        metadata_file_loader: AssetMetadataFileLoader,
    ) -> None:
        self._file_lister = file_lister
        self._metadata_file_loader = metadata_file_loader

    @override
    def load(self, package: str) -> AssetMetadataProvider:
        source = f"package:{package}"

        metadata = {}

        for file in self._file_lister.list(package, source):
            if file.suffix != ".yaml" and file.suffix != ".yml":
                continue

            file_metadata = self._metadata_file_loader.load(file, source)

            for name, asset_metadata in file_metadata:
                if name in metadata:
                    msg = f"Two assets in the {package} package have the same name {name}."

                    raise AssetMetadataError(source, msg)

                metadata[name] = asset_metadata

        return CachedAssetMetadataProvider(source, metadata)


class PackageFileLister(ABC):
    @abstractmethod
    def list(self, package: str, source: str) -> list[Path]: ...


@final
class StandardPackageFileLister(PackageFileLister):
    @override
    def list(self, package: str, source: str) -> list[Path]:
        files = []

        def collect_files(p: MultiplexedPath | Path) -> None:
            if p.is_file():
                if not isinstance(p, Path):
                    raise OperationalError(
                        f"`importlib.resources` returned a path of type `{type(p)}`."
                    )

                files.append(p)
            elif p.is_dir():
                for e in p.iterdir():
                    collect_files(e)

        try:
            path = get_files(package)
        except ModuleNotFoundError:
            raise AssetSourceNotFoundError(source) from None
        except RuntimeError as ex:
            raise OperationalError(
                f"Assets in the {package} package cannot be retrieved."
            ) from ex

        collect_files(path)

        return files


def load_in_memory_asset_metadata(
    source: str, entries: Sequence[dict[str, object]]
) -> AssetMetadataProvider:
    metadata = {}

    for idx, asset_metadata in enumerate(entries):
        name = asset_metadata.pop("name", None)
        if not name:
            msg = f"Asset at index {idx} in `metadata` must have a name."

            raise AssetMetadataError(source, msg)

        if not isinstance(name, str):
            msg = f"Name of the asset at index {idx} in `metadata` must be of type `{str}`, but is of type `{type(name)}` instead."

            raise AssetMetadataError(source, msg)

        name = canonicalize_asset_name(name)
        if name is None:
            msg = f"Asset at index {idx} in `metadata` must have a valid name."

            raise AssetMetadataError(source, msg)

        if name in metadata:
            msg = f"Assets in `metadata` must have unique names, but two assets have the same name {name}."

            raise AssetMetadataError(source, msg)

        base_name = asset_metadata.get("base")
        if base_name is not None:
            if not isinstance(base_name, str):
                msg = f"Base name of the asset at index {idx} in `metadata` must be of type `{str}`, but is of type `{type(base_name)}` instead."

                raise AssetMetadataError(source, msg)

            base_name = sanitize_base_asset_name(base_name)
            if base_name is None:
                msg = f"Asset at index {idx} in `metadata` must have a valid base name."

                raise AssetMetadataError(source, msg)

        metadata[name] = asset_metadata

    return CachedAssetMetadataProvider(source, metadata)


class AssetMetadataFileLoader(ABC):
    @abstractmethod
    def load(self, file: Path, source: str) -> list[tuple[str, dict[str, object]]]:
        """Load asset metadata included in ``file``."""


@final
class YamlAssetMetadataFileLoader(AssetMetadataFileLoader):
    def __init__(self, yaml_loader: YamlLoader) -> None:
        self._yaml_loader = yaml_loader

    @override
    def load(self, file: Path, source: str) -> list[tuple[str, dict[str, object]]]:
        output = []

        try:
            file_metadata = self._yaml_loader.load(file)
        except YamlError as ex:
            msg = f"{file} cannot be loaded as YAML."

            raise AssetMetadataError(source, msg) from ex

        for idx, asset_metadata in enumerate(file_metadata):
            if not isinstance(asset_metadata, dict):
                msg = f"Metadata of the asset at index {idx} in {file} is expected to be of type `{dict}`, but is of type `{type(asset_metadata)}` instead."

                raise AssetMetadataError(source, msg)

            name = asset_metadata.pop("name", None)
            if not name:
                msg = f"Asset at index {idx} in {file} does not have a name."

                raise AssetMetadataError(source, msg)

            if not isinstance(name, str):
                msg = f"Name of the asset at index {idx} in {file} is expected to be of type `{str}`, but is of type `{type(name)}` instead."

                raise AssetMetadataError(source, msg)

            name = canonicalize_asset_name(name)
            if name is None:
                msg = f"Asset at index {idx} in {file} does not have a valid name."

                raise AssetMetadataError(source, msg)

            base_name = asset_metadata.get("base")
            if base_name is not None:
                if not isinstance(base_name, str):
                    msg = f"Base name of the asset at index {idx} in {file} is expected to be of type `{str}`, but is of type `{type(base_name)}` instead."

                    raise AssetMetadataError(source, msg)

                base_name = sanitize_base_asset_name(base_name)
                if base_name is None:
                    msg = f"Asset at index {idx} in {file} does not have a valid base name."

                    raise AssetMetadataError(source, msg)

                asset_metadata["base"] = base_name

            asset_metadata["__base_path__"] = file.parent

            output.append((name, asset_metadata))

        return output


def canonicalize_asset_name(name: str) -> str | None:
    name_env_pair = name.split("@")

    match name_env_pair:
        case [name]:
            name = f"{name.strip()}@"
        case [name, env]:
            name = f"{name.strip()}@{env.strip()}"
        case _:
            return None

    if name[0] == "@":
        return None

    return name


def sanitize_base_asset_name(name: str) -> str | None:
    if "@" in name:
        return None

    name = name.strip()
    if not name:
        return None

    return name


@final
class WellKnownAssetMetadataSource(AssetMetadataSource):
    def __init__(
        self, dirs: AssetDirectoryAccessor, metadata_loader: FileAssetMetadataLoader
    ) -> None:
        self._dirs = dirs
        self._metadata_loader = metadata_loader

    @override
    def load(self) -> Iterator[AssetMetadataProvider]:
        try:
            path = self._dirs.maybe_get_system_dir()
            if path is not None:
                try:
                    yield self._metadata_loader.load(path)
                except AssetSourceNotFoundError:
                    pass

            path = self._dirs.maybe_get_user_dir()
            if path is not None:
                try:
                    yield self._metadata_loader.load(path)
                except AssetSourceNotFoundError:
                    pass
        except OSError as ex:
            raise_operational_system_error(ex)


@final
class FileAssetMetadataSource(AssetMetadataSource):
    def __init__(
        self, path: Path, metadata_loader: FileAssetMetadataLoader, not_exist_ok: bool
    ) -> None:
        self._path = path
        self._metadata_loader = metadata_loader
        self._not_exist_ok = not_exist_ok

    @override
    def load(self) -> Iterator[AssetMetadataProvider]:
        try:
            yield self._metadata_loader.load(self._path)
        except AssetSourceNotFoundError:
            if self._not_exist_ok:
                return

            raise
        except OSError as ex:
            raise_operational_system_error(ex)


@final
class PackageAssetMetadataSource(AssetMetadataSource):
    def __init__(
        self, package: str, metadata_loader: PackageAssetMetadataLoader
    ) -> None:
        self._package = package
        self._metadata_loader = metadata_loader

    @override
    def load(self) -> Iterator[AssetMetadataProvider]:
        try:
            yield self._metadata_loader.load(self._package)
        except OSError as ex:
            raise_operational_system_error(ex)


@final
class InMemoryAssetMetadataSource(AssetMetadataSource):
    def __init__(self, name: str, entries: Sequence[dict[str, object]]) -> None:
        self._name = name
        self._entries = entries

    @override
    def load(self) -> Iterator[AssetMetadataProvider]:
        yield load_in_memory_asset_metadata(self._name, self._entries)
