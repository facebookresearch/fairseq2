# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence, Set
from copy import deepcopy
from functools import partial
from pathlib import Path
from pickle import PickleError
from typing import NoReturn, final

from importlib_resources import files as get_files
from importlib_resources.readers import MultiplexedPath
from typing_extensions import override

from fairseq2.error import ContractError, FormatError, InfraError, InternalError
from fairseq2.file_system import FileSystem
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver
from fairseq2.utils.yaml import YamlError, YamlLoader


class AssetMetadataProvider(ABC):
    """Provides asset metadata."""

    @abstractmethod
    def get_metadata(self, name: str) -> dict[str, object]:
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


@final
class CachedAssetMetadataProvider(AssetMetadataProvider):
    _source: str
    _metadata: dict[str, dict[str, object]]

    def __init__(self, source: str, metadata: dict[str, dict[str, object]]) -> None:
        self._source = source
        self._metadata = metadata

    @override
    def get_metadata(self, name: str) -> dict[str, object]:
        try:
            asset_metadata = self._metadata[name]
        except KeyError:
            raise AssetNotFoundError(name) from None

        try:
            return deepcopy(asset_metadata)
        except (RuntimeError, TypeError, ValueError, PickleError) as ex:
            raise ContractError(
                f"The metadata of the '{name}' asset cannot be copied. See the nested exception for details."
            ) from ex

    @property
    @override
    def asset_names(self) -> Set[str]:
        return self._metadata.keys()

    @property
    @override
    def source(self) -> str:
        return self._source


@final
class FileBackedAssetMetadataLoader:
    """Provides asset metadata stored on a file system."""

    _file_system: FileSystem
    _metadata_file_loader: AssetMetadataFileLoader

    def __init__(
        self,
        file_system: FileSystem,
        metadata_file_loader: AssetMetadataFileLoader,
    ) -> None:
        self._file_system = file_system
        self._metadata_file_loader = metadata_file_loader

    def load(self, path: Path) -> AssetMetadataProvider:
        metadata = {}

        def cache_file(file: Path, source: str) -> None:
            try:
                file_metadata = self._metadata_file_loader.load(file, source)
            except FileNotFoundError as ex:
                if file == path:
                    raise

                raise InfraError(f"The '{file}' file is not found.") from ex

            for name, asset_metadata in file_metadata:
                if name in metadata:
                    if file == path:
                        raise AssetMetadataError(
                            source, f"Two assets in the '{path}' file have the same name '{name}'."  # fmt: skip
                        )
                    else:
                        raise AssetMetadataError(
                            source, f"Two assets under the '{path}' directory have the same name '{name}'."  # fmt: skip
                        )

                metadata[name] = asset_metadata

        try:
            is_dir = self._file_system.is_dir(path)
        except OSError as ex:
            raise InfraError(
                f"A system error has occurred while accessing the '{path}' path. See the nested exception for details."
            ) from ex

        if is_dir:
            source = f"directory:{path}"

            def on_error(ex: OSError) -> NoReturn:
                raise InfraError(
                    f"A system error has occurred while traversing the '{path}' directory. See the nested exception for details."
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
            source = f"file:{path}"

            cache_file(path, source)

        return CachedAssetMetadataProvider(source, metadata)


@final
class PackageBackedAssetMetadataLoader:
    """Provides asset metadata stored in a Python namespace package."""

    _package_name: str
    _file_lister: PackageFileLister
    _metadata_file_loader: AssetMetadataFileLoader

    def __init__(
        self,
        package_name: str,
        file_lister: PackageFileLister,
        metadata_file_loader: AssetMetadataFileLoader,
    ) -> None:
        self._package_name = package_name
        self._file_lister = file_lister
        self._metadata_file_loader = metadata_file_loader

    def load(self) -> AssetMetadataProvider:
        source = f"package:{self._package_name}"

        metadata = {}

        for file in self._file_lister.list(self._package_name):
            if file.suffix != ".yaml" and file.suffix != ".yml":
                continue

            try:
                file_metadata = self._metadata_file_loader.load(file, source)
            except FileNotFoundError as ex:
                raise InfraError(
                    f"The '{self._package_name}' package does not have a file named '{file}'."
                ) from ex

            for name, asset_metadata in file_metadata:
                if name in metadata:
                    raise AssetMetadataError(
                        source, f"Two assets in the '{self._package_name}' package have the same name '{name}'."  # fmt: skip
                    )

                metadata[name] = asset_metadata

        return CachedAssetMetadataProvider(source, metadata)


def load_in_mem_asset_metadata_provider(
    entries: Sequence[dict[str, object]],
) -> AssetMetadataProvider:
    source = "mem"

    metadata = {}

    for idx, asset_metadata in enumerate(entries):
        try:
            raw_name = asset_metadata.pop("name")
        except KeyError:
            raise AssetMetadataError(
                source, f"The asset at index {idx} in `metadata` does not have a name."
            ) from None

        try:
            name = canonicalize_asset_name(raw_name)
        except (TypeError, FormatError) as ex:
            raise AssetMetadataError(
                source, f"The asset at index {idx} in `metadata` does not have a valid name. See the nested exception for details."  # fmt: skip
            ) from ex

        if name in metadata:
            raise AssetMetadataError(
                source, f"Two assets in `metadata` have the same name '{name}'."
            )

        base_name = asset_metadata.get("base")
        if base_name is not None:
            try:
                asset_metadata["base"] = sanitize_base_asset_name(base_name)
            except (TypeError, FormatError) as ex:
                raise AssetMetadataError(
                    source, f"The asset at index {idx} in `metadata` does not have a valid base name. See the nested exception for details."  # fmt: skip
                ) from ex

        metadata[name] = asset_metadata

    return CachedAssetMetadataProvider(source, metadata)


class AssetMetadataFileLoader(ABC):
    @abstractmethod
    def load(self, file: Path, source: str) -> list[tuple[str, dict[str, object]]]:
        """Load asset metadata included in ``file``."""


@final
class YamlAssetMetadataFileLoader(AssetMetadataFileLoader):
    _yaml_loader: YamlLoader

    def __init__(self, yaml_loader: YamlLoader) -> None:
        self._yaml_loader = yaml_loader

    @override
    def load(self, file: Path, source: str) -> list[tuple[str, dict[str, object]]]:
        output = []

        try:
            file_metadata = self._yaml_loader.load(file)
        except FileNotFoundError:
            raise
        except (OSError, YamlError) as ex:
            raise InfraError(
                f"The '{file}' file cannot be loaded as YAML. See the nested exception for details."
            ) from ex

        for idx, asset_metadata in enumerate(file_metadata):
            if not isinstance(asset_metadata, dict):
                raise AssetMetadataError(
                    source, f"The metadata of the asset at index {idx} in the '{file}' file is expected to be of type `{dict}`, but is of type `{type(asset_metadata)}` instead."  # fmt: skip
                )

            try:
                raw_name = asset_metadata.pop("name")
            except KeyError:
                raise AssetMetadataError(
                    source, f"The asset at index {idx} in the '{file}' file does not have a name."  # fmt: skip
                ) from None

            try:
                name = canonicalize_asset_name(raw_name)
            except (TypeError, FormatError) as ex:
                raise AssetMetadataError(
                    source, f"The asset at index {idx} in the '{file}' file does not have a valid name. See the nested exception for details."  # fmt: skip
                ) from ex

            base_name = asset_metadata.get("base")
            if base_name is not None:
                try:
                    asset_metadata["base"] = sanitize_base_asset_name(base_name)
                except (TypeError, FormatError) as ex:
                    raise AssetMetadataError(
                        source, f"The asset at index {idx} in the '{file}' file does not have a valid base name. See the nested exception for details"  # fmt: skip
                    ) from ex

            asset_metadata["__base_path__"] = file.parent

            output.append((name, asset_metadata))

        return output


class PackageFileLister(ABC):
    @abstractmethod
    def list(self, package_name: str) -> list[Path]: ...


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


def canonicalize_asset_name(name: object) -> str:
    if not isinstance(name, str):
        raise TypeError(
            f"`name` must be of type `{str}`, but is of type `{type(name)}` instead."
        )

    name_env_pair = name.split("@")

    match name_env_pair:
        case [name]:
            return f"{name.strip()}@"
        case [name, env]:
            return f"{name.strip()}@{env.strip()}"
        case _:
            raise FormatError(
                "'@' is a reserved character and must not occur more than once in `name`."
            )


def sanitize_base_asset_name(name: object) -> str:
    if not isinstance(name, str):
        raise TypeError(
            f"`name` must be of type `{str}`, but is of type `{type(name)}` instead."
        )

    if "@" in name:
        raise FormatError("'@' is a reserved character and must not occur in `name`.")

    return name.strip()


class AssetNotFoundError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"An asset with name '{name}' is not found.")

        self.name = name


class AssetMetadataError(Exception):
    source: str

    def __init__(self, source: str, message: str) -> None:
        super().__init__(message)

        self.source = source


def register_assets(
    container: DependencyContainer, path: Path, *, not_exist_ok: bool = False
) -> None:
    def load_assets(resolver: DependencyResolver) -> AssetMetadataProvider | None:
        file_system = resolver.resolve(FileSystem)

        yaml_loader = resolver.resolve(YamlLoader)

        metadata_file_loader = YamlAssetMetadataFileLoader(yaml_loader)

        metadata_loader = FileBackedAssetMetadataLoader(
            file_system, metadata_file_loader
        )

        try:
            return metadata_loader.load(path)
        except FileNotFoundError as ex:
            if not_exist_ok:
                return None

            raise AssetMetadataError(
                f"path:{path}", f"The '{path}' path is not found."
            ) from ex

    container.register(AssetMetadataProvider, load_assets)


def register_package_assets(container: DependencyContainer, package_name: str) -> None:
    loader = partial(_load_package_assets, package_name=package_name)

    container.register(AssetMetadataProvider, loader)


def _load_package_assets(
    resolver: DependencyResolver, package_name: str
) -> AssetMetadataProvider:
    yaml_loader = resolver.resolve(YamlLoader)

    file_lister = WheelPackageFileLister()

    metadata_file_loader = YamlAssetMetadataFileLoader(yaml_loader)

    metadata_loader = PackageBackedAssetMetadataLoader(
        package_name, file_lister, metadata_file_loader
    )

    return metadata_loader.load()
