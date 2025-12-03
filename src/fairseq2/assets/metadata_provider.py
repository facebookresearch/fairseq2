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

from fairseq2.assets.dirs import AssetDirectoryError, _AssetDirectoryAccessor
from fairseq2.error import InternalError
from fairseq2.file_system import FileSystem
from fairseq2.utils.yaml import YamlError, YamlLoader


class AssetMetadataProvider(ABC):
    """Provides asset metadata."""

    @abstractmethod
    def maybe_get_metadata(self, name: str) -> dict[str, object] | None:
        """Returns the metadata of the specified asset."""

    @property
    @abstractmethod
    def asset_names(self) -> Set[str]:
        """Gets the names of the assets for which this provider has metadata."""

    @property
    @abstractmethod
    def source(self) -> str: ...


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
            raise InternalError(
                f"failed to deepcopy metadata of asset '{name}' in asset metadata source '{self._source}'"
            ) from ex

    @property
    @override
    def asset_names(self) -> Set[str]:
        return self._metadata.keys()

    @property
    @override
    def source(self) -> str:
        return self._source


class AssetMetadataSource(ABC):
    """Represents source of one or more asset metadata providers."""

    @abstractmethod
    def load(self) -> Iterator[AssetMetadataProvider]:
        """
        :raises AssetMetadataSourceNotFoundError:
        :raises BadAssetMetadataError:
        :raises AssetMetadataError:
        """


class AssetMetadataError(Exception):
    def __init__(self, source: str, message: str) -> None:
        super().__init__(message)

        self.source = source


class AssetMetadataSourceNotFoundError(AssetMetadataError):
    def __init__(self, source: str) -> None:
        super().__init__(source, f"asset metadata source '{source}' is not found")


class BadAssetMetadataError(AssetMetadataError):
    pass


@final
class _WellKnownAssetMetadataSource(AssetMetadataSource):
    def __init__(
        self, dirs: _AssetDirectoryAccessor, metadata_loader: FileAssetMetadataLoader
    ) -> None:
        self._dirs = dirs
        self._metadata_loader = metadata_loader

    @override
    def load(self) -> Iterator[AssetMetadataProvider]:
        source = "system"

        try:
            path = self._dirs.maybe_get_system_dir()
        except AssetDirectoryError as ex:
            raise AssetMetadataError(
                source, f"failed to determine the path of asset metadata source '{source}'"  # fmt: skip
            ) from ex

        if path is not None:
            try:
                yield self._metadata_loader.load(source, path)
            except AssetMetadataSourceNotFoundError:
                pass

        source = "user"

        try:
            path = self._dirs.maybe_get_user_dir()
        except AssetDirectoryError as ex:
            raise AssetMetadataError(
                source, f"failed to determine the path of asset metadata source '{source}'"  # fmt: skip
            ) from ex

        if path is not None:
            try:
                yield self._metadata_loader.load(source, path)
            except AssetMetadataSourceNotFoundError:
                pass


@final
class _FileAssetMetadataSource(AssetMetadataSource):
    def __init__(
        self, path: Path, metadata_loader: FileAssetMetadataLoader, not_exist_ok: bool
    ) -> None:
        self._path = path
        self._metadata_loader = metadata_loader
        self._not_exist_ok = not_exist_ok

    @override
    def load(self) -> Iterator[AssetMetadataProvider]:
        source = f"path:{self._path}"

        try:
            yield self._metadata_loader.load(source, self._path)
        except AssetMetadataSourceNotFoundError:
            if self._not_exist_ok:
                return

            raise


@final
class _PackageAssetMetadataSource(AssetMetadataSource):
    def __init__(
        self, package: str, metadata_loader: _PackageAssetMetadataLoader
    ) -> None:
        self._package = package
        self._metadata_loader = metadata_loader

    @override
    def load(self) -> Iterator[AssetMetadataProvider]:
        source = f"package:{self._package}"

        yield self._metadata_loader.load(source, self._package)


@final
class _InMemoryAssetMetadataSource(AssetMetadataSource):
    def __init__(self, name: str, entries: Sequence[dict[str, object]]) -> None:
        self._name = name
        self._entries = entries

    @override
    def load(self) -> Iterator[AssetMetadataProvider]:
        yield _load_in_memory_asset_metadata(self._name, self._entries)


class FileAssetMetadataLoader(ABC):
    """Loads asset metadata stored on a file system."""

    @abstractmethod
    def load(self, source: str, path: Path) -> AssetMetadataProvider:
        """
        :raises AssetMetadataSourceNotFoundError:
        :raises BadAssetMetadataError:
        :raises AssetMetadataError:
        """


@final
class _StandardFileAssetMetadataLoader(FileAssetMetadataLoader):
    def __init__(
        self,
        file_system: FileSystem,
        metadata_file_loader: _AssetMetadataFileLoader,
    ) -> None:
        self._file_system = file_system
        self._metadata_file_loader = metadata_file_loader

    @override
    def load(self, source: str, path: Path) -> AssetMetadataProvider:
        try:
            path = self._file_system.resolve(path)
        except RuntimeError as ex:
            raise AssetMetadataError(source, f"failed to access path '{path}'") from ex

        try:
            is_dir = self._file_system.is_dir(path)
        except OSError as ex:
            raise AssetMetadataError(source, f"failed to access path '{path}'") from ex

        if is_dir:
            metadata = {}

            def on_error(ex: OSError) -> NoReturn:
                raise AssetMetadataError(
                    source, f"failed to walk directory '{path}'"
                ) from ex

            for dir_pathname, filenames in self._file_system.walk_directory(
                path, on_error=on_error
            ):
                metadata_dir = Path(dir_pathname)

                for filename in filenames:
                    file = metadata_dir.joinpath(filename)

                    if file.suffix != ".yaml" and file.suffix != ".yml":
                        continue

                    try:
                        file_metadata = self._metadata_file_loader.load(file)
                    except BadAssetMetadataFileError as ex:
                        raise BadAssetMetadataError(
                            source, f"failed to load asset metadata file '{file}'"
                        ) from ex
                    except FileNotFoundError:
                        continue
                    except OSError as ex:
                        raise AssetMetadataError(
                            source, f"failed to read file '{file}'"
                        ) from ex

                    for name, asset_metadata in file_metadata:
                        if name in metadata:
                            raise BadAssetMetadataError(
                                source, f"two assets under directory '{path}' have the same name '{name}'"  # fmt: skip
                            )

                        metadata[name] = asset_metadata
        else:
            try:
                file_metadata = self._metadata_file_loader.load(path)
            except BadAssetMetadataFileError as ex:
                raise BadAssetMetadataError(
                    source, f"failed to load asset metadata file '{path}'"
                ) from ex
            except FileNotFoundError:
                raise AssetMetadataSourceNotFoundError(source) from None
            except OSError as ex:
                raise AssetMetadataError(
                    source, f"failed to read file '{path}'"
                ) from ex

            metadata = dict(file_metadata)

        return CachedAssetMetadataProvider(source, metadata)


class _PackageAssetMetadataLoader(ABC):
    """Provides asset metadata stored in a Python namespace package."""

    @abstractmethod
    def load(self, source: str, package: str) -> AssetMetadataProvider:
        """
        :raises AssetMetadataSourceNotFoundError:
        :raises BadAssetMetadataError:
        :raises AssetMetadataError:
        """


@final
class _StandardPackageAssetMetadataLoader(_PackageAssetMetadataLoader):
    def __init__(
        self,
        file_lister: _PackageFileLister,
        metadata_file_loader: _AssetMetadataFileLoader,
    ) -> None:
        self._file_lister = file_lister
        self._metadata_file_loader = metadata_file_loader

    @override
    def load(self, source: str, package: str) -> AssetMetadataProvider:
        metadata = {}

        try:
            files = self._file_lister.list(package)
        except ModuleNotFoundError:
            raise AssetMetadataSourceNotFoundError(source) from None
        except OSError as ex:
            raise AssetMetadataError(
                source, f"failed to list asset metadata files of package '{package}'"
            ) from ex

        for file in files:
            if file.suffix != ".yaml" and file.suffix != ".yml":
                continue

            try:
                file_metadata = self._metadata_file_loader.load(file)
            except BadAssetMetadataFileError as ex:
                raise BadAssetMetadataError(
                    source, f"failed to load asset metadata file '{file}'"
                ) from ex
            except OSError as ex:
                raise AssetMetadataError(
                    source, f"failed to read file '{file}'"
                ) from ex

            for name, asset_metadata in file_metadata:
                if name in metadata:
                    raise BadAssetMetadataError(
                        source, f"two assets in package '{package}' have the same name '{name}'"  # fmt: skip
                    )

                metadata[name] = asset_metadata

        return CachedAssetMetadataProvider(source, metadata)


class _PackageFileLister(ABC):
    @abstractmethod
    def list(self, package: str) -> list[Path]:
        """
        :raises ModuleNotFoundError:
        :raises OSError:
        """


@final
class _StandardPackageFileLister(_PackageFileLister):
    @override
    def list(self, package: str) -> list[Path]:
        files = []

        def collect_files(p: MultiplexedPath | Path) -> None:
            if p.is_file():
                if not isinstance(p, Path):
                    raise InternalError(
                        f"`importlib.resources` returned a path of type `{type(p)}`"
                    )

                files.append(p)
            elif p.is_dir():
                for e in p.iterdir():
                    collect_files(e)

        path = get_files(package)

        collect_files(path)

        return files


def _load_in_memory_asset_metadata(
    source: str, entries: Sequence[dict[str, object]]
) -> AssetMetadataProvider:
    """
    :raises BadAssetMetadataError:
    """
    metadata = {}

    for idx, asset_metadata in enumerate(entries):
        name = asset_metadata.pop("name", None)
        if not name:
            raise BadAssetMetadataError(
                source, f"asset at index {idx} does not have a name"
            )

        if not isinstance(name, str):
            raise BadAssetMetadataError(
                source, f"name of the asset at index {idx} is expected to be of type `{str}`, but is of type `{type(name)}` instead"  # fmt: skip
            )

        name = canonicalize_asset_name(name)
        if name is None:
            raise BadAssetMetadataError(
                source, f"asset at index {idx} does not have a valid name '{name}'."
            )

        if name in metadata:
            raise BadAssetMetadataError(
                source, f"two assets have the same name '{name}'"
            )

        base_name = asset_metadata.get("base")
        if base_name is not None:
            if not isinstance(base_name, str):
                raise BadAssetMetadataError(
                    source, f"base name of the asset at index {idx} must be of type `{str}`, but is of type `{type(base_name)}` instead"  # fmt: skip
                )

            base_name = sanitize_base_asset_name(base_name)
            if base_name is None:
                raise BadAssetMetadataError(
                    source, f"asset at index {idx} does not have a valid base name '{base_name}'"  # fmt: skip
                )

        metadata[name] = asset_metadata

    return CachedAssetMetadataProvider(source, metadata)


class _AssetMetadataFileLoader(ABC):
    @abstractmethod
    def load(self, file: Path) -> list[tuple[str, dict[str, object]]]:
        """
        :raises BadAssetMetadataFileError:
        :raises FileNotFoundError:
        :raises OSError:
        """


class BadAssetMetadataFileError(Exception):
    def __init__(self, file: Path, message: str) -> None:
        super().__init__(message)

        self.file = file


@final
class _YamlAssetMetadataFileLoader(_AssetMetadataFileLoader):
    def __init__(self, yaml_loader: YamlLoader) -> None:
        self._yaml_loader = yaml_loader

    @override
    def load(self, file: Path) -> list[tuple[str, dict[str, object]]]:
        metadata = []

        try:
            file_metadata = self._yaml_loader.load(file)
        except YamlError as ex:
            raise BadAssetMetadataFileError(
                file, f"failed to load file '{file}' as YAML"
            ) from ex

        for idx, asset_metadata in enumerate(file_metadata):
            if not isinstance(asset_metadata, dict):
                raise BadAssetMetadataFileError(
                    file, f"metadata of the asset at index {idx} in file '{file}' is expected to be of type `{dict}`, but is of type `{type(asset_metadata)}` instead"  # fmt: skip
                )

            name = asset_metadata.pop("name", None)
            if not name:
                raise BadAssetMetadataFileError(
                    file, f"asset at index {idx} in file '{file}' does not have a name"
                )

            if not isinstance(name, str):
                raise BadAssetMetadataFileError(
                    file, f"name of the asset at index {idx} in file '{file}' is expected to be of type `{str}`, but is of type `{type(name)}` instead"  # fmt: skip
                )

            name = canonicalize_asset_name(name)
            if name is None:
                raise BadAssetMetadataFileError(
                    file, f"asset at index {idx} in file '{file}' does not have a valid name '{name}'"  # fmt: skip
                )

            if name in metadata:
                raise BadAssetMetadataFileError(
                    file, f"two assets in file '{file}' have the same name '{name}'"
                )

            base_name = asset_metadata.get("base")
            if base_name is not None:
                if not isinstance(base_name, str):
                    raise BadAssetMetadataFileError(
                        file, f"base name of the asset at index {idx} in file '{file}' is expected to be of type `{str}`, but is of type `{type(base_name)}` instead"  # fmt: skip
                    )

                base_name = sanitize_base_asset_name(base_name)
                if base_name is None:
                    raise BadAssetMetadataFileError(
                        file, f"asset at index {idx} in file '{file}' does not have a valid base name '{base_name}'"  # fmt: skip
                    )

                asset_metadata["base"] = base_name

            asset_metadata["__base_path__"] = file.parent

            metadata.append((name, asset_metadata))

        return metadata


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
