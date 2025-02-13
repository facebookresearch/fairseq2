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


@final
class CachedAssetMetadataProvider(AssetMetadataProvider):
    _metadata: dict[str, dict[str, object]]

    def __init__(self, metadata: dict[str, dict[str, object]]) -> None:
        self._metadata = metadata

    @override
    def get_metadata(self, name: str) -> dict[str, object]:
        try:
            metadata = self._metadata[name]
        except KeyError:
            raise AssetMetadataNotFoundError(name) from None

        try:
            return deepcopy(metadata)
        except Exception as ex:
            raise ContractError(
                f"The metadata of the '{name}' asset cannot be copied. See the nested exception for details."
            ) from ex

    @override
    def get_names(self) -> list[str]:
        return list(self._metadata.keys())


@final
class FileAssetMetadataLoader:
    """Provides asset metadata stored on a file system."""

    _path: Path
    _file_system: FileSystem
    _metadata_file_loader: AssetMetadataFileLoader

    def __init__(
        self,
        path: Path,
        file_system: FileSystem,
        metadata_file_loader: AssetMetadataFileLoader,
    ) -> None:
        super().__init__()

        self._path = path
        self._file_system = file_system
        self._metadata_file_loader = metadata_file_loader

    def load(self) -> AssetMetadataProvider:
        path = self._path

        cache = {}

        def cache_file(file: Path, source: str) -> None:
            try:
                all_metadata = self._metadata_file_loader.load(file)
            except FileNotFoundError:
                if file == path:
                    raise

                raise AssetMetadataLoadError(
                    f"The '{file}' file is not found."
                ) from None

            for name, metadata in all_metadata:
                if name in cache:
                    if file == path:
                        raise AssetMetadataLoadError(
                            f"Two assets in the '{path}' file have the same name '{name}'."
                        )
                    else:
                        raise AssetMetadataLoadError(
                            f"Two assets under the '{path}' directory have the same name '{name}'."
                        )

                metadata["__source__"] = source

                cache[name] = metadata

        try:
            is_dir = self._file_system.is_dir(path)
        except OSError as ex:
            raise AssetMetadataLoadError(
                f"The '{path}' path cannot be accessed. See the nested exception for details."
            ) from ex

        if is_dir:
            source = f"directory:{path}"

            def on_error(ex: OSError) -> NoReturn:
                raise AssetMetadataLoadError(
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

        return CachedAssetMetadataProvider(cache)


@final
class PackageAssetMetadataLoader:
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
        super().__init__()

        self._package_name = package_name
        self._file_lister = file_lister
        self._metadata_file_loader = metadata_file_loader

    def load(self) -> AssetMetadataProvider:
        source = f"package:{self._package_name}"

        cache = {}

        for file in self._file_lister.list(self._package_name):
            if file.suffix != ".yaml" and file.suffix != ".yml":
                continue

            try:
                all_metadata = self._metadata_file_loader.load(file)
            except FileNotFoundError:
                raise AssetMetadataLoadError(
                    f"The '{self._package_name}' package does not have a file named '{file}'."
                ) from None

            for name, metadata in all_metadata:
                if name in cache:
                    raise AssetMetadataLoadError(
                        f"Two assets in the '{self._package_name}' package have the same name '{name}'."
                    )

                metadata["__source__"] = source

                cache[name] = metadata

        return CachedAssetMetadataProvider(cache)


class AssetMetadataFileLoader(ABC):
    @abstractmethod
    def load(self, file: Path) -> list[tuple[str, dict[str, object]]]:
        """Load asset metadata included in ``file``."""


@final
class StandardAssetMetadataFileLoader(AssetMetadataFileLoader):
    _yaml_loader: YamlLoader

    def __init__(self, yaml_loader: YamlLoader) -> None:
        self._yaml_loader = yaml_loader

    @override
    def load(self, file: Path) -> list[tuple[str, dict[str, object]]]:
        output = []

        try:
            all_metadata = self._yaml_loader.load(file)
        except FileNotFoundError:
            raise
        except (OSError, YamlError) as ex:
            raise AssetMetadataLoadError(
                f"The '{file}' asset metadata file cannot be loaded as YAML. See the nested exception for details."
            ) from ex

        for idx, metadata in enumerate(all_metadata):
            if not isinstance(metadata, dict):
                raise AssetMetadataLoadError(
                    f"The asset metadata at index {idx} in the '{file}' file is expected to be of type `dict`, but is of type `{type(metadata)}` instead."
                )

            try:
                name = metadata.pop("name")
            except KeyError:
                raise AssetMetadataLoadError(
                    f"The asset metadata at index {idx} in the '{file}' file does not have a name."
                ) from None

            try:
                canonical_name = _canonicalize_name(name)
            except ValueError as ex:
                raise AssetMetadataLoadError(
                    f"The asset metadata at index {idx} in the '{file}' file does not have a valid name. See the nested exception for details."
                ) from ex

            base = metadata.get("base")
            if base is not None:
                if not isinstance(base, str) or "@" in base:
                    raise AssetMetadataLoadError(
                        f"The asset metadata at index {idx} in the '{file}' file does not have a valid base name."
                    )

            metadata["__base_path__"] = file.parent

            output.append((canonical_name, metadata))

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


@final
class InProcAssetMetadataLoader:
    """Provides asset metadata stored in memory."""

    _metadata: Sequence[dict[str, object]]

    def __init__(self, metadata: Sequence[dict[str, object]]) -> None:
        self._metadata = metadata

    def load(self) -> AssetMetadataProvider:
        cache = {}

        for idx, metadata in enumerate(self._metadata):
            try:
                name = metadata.pop("name")
            except KeyError:
                raise AssetMetadataLoadError(
                    f"The asset metadata at index {idx} in `metadata` does not have a name."
                ) from None

            try:
                canonical_name = _canonicalize_name(name)
            except ValueError as ex:
                raise AssetMetadataLoadError(
                    f"The asset metadata at index {idx} in `metadata` does not have a valid name. See the nested exception for details."
                ) from ex

            if canonical_name in cache:
                raise AssetMetadataLoadError(
                    f"Two assets in `metadata` have the same name '{canonical_name}'."
                )

            base = metadata.get("base")
            if base is not None:
                if not isinstance(base, str) or "@" in base:
                    raise AssetMetadataLoadError(
                        f"The asset metadata at index {idx} in `metadata` file does not have a valid base name."
                    )

            metadata["__source__"] = "inproc"

            cache[canonical_name] = metadata

        return CachedAssetMetadataProvider(cache)


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


class AssetMetadataNotFoundError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"An asset metadata with name '{name}' is not found.")

        self.name = name


class AssetMetadataLoadError(Exception):
    pass


class AssetMetadataSaveError(Exception):
    pass
