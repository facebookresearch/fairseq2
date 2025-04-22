# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from itertools import count
from pathlib import Path
from pickle import PickleError
from typing import TypeAlias, cast, final

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.error import NotSupportedError
from fairseq2.file_system import FileMode, FileSystem
from fairseq2.typing import Device

MapLocation: TypeAlias = (
    Callable[[Tensor, str], Tensor] | Device | str | dict[str, str] | None
)


class TensorLoader(ABC):
    """Loads tensors from files."""

    @abstractmethod
    def load(
        self,
        path: Path,
        *,
        map_location: MapLocation = None,
        restrict: bool = True,
        mmap: bool = False,
    ) -> dict[str, object]:
        """
        :param path:
            The path to the file.
        :param map_location:
            Same as the ``map_location`` parameter of :meth:`torch.load`.
        """


class TensorDumper(ABC):
    """Dumps tensors to files."""

    @abstractmethod
    def dump(self, data: Mapping[str, object], path: Path) -> None:
        """
        :param data:
            The dictionary containing tensors and other auxiliary data.
        :param path:
            The path to the file.
        """


@final
class TorchTensorLoader(TensorLoader):
    _file_system: FileSystem

    def __init__(self, file_system: FileSystem) -> None:
        self._file_system = file_system

    @override
    def load(
        self,
        path: Path,
        *,
        map_location: MapLocation = None,
        restrict: bool = True,
        mmap: bool = False,
    ) -> dict[str, object]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message=r".*You are using `torch\.load` with `weights_only=False`.*"  # fmt: skip
            )

            def load_error() -> TensorLoadError:
                return TensorLoadError(
                    path, f"The '{path}' tensor file cannot be loaded. See the nested exception for details."  # fmt: skip
                )

            try:
                data: dict[str, object] = torch.load(
                    path, map_location, weights_only=restrict, mmap=mmap  # type: ignore[arg-type]
                )
            except FileNotFoundError:
                raise
            except (RuntimeError, OSError, PickleError) as ex:
                raise load_error() from ex

        return data


@final
class TorchTensorDumper(TensorDumper):
    _file_system: FileSystem

    def __init__(self, file_system: FileSystem) -> None:
        self._file_system = file_system

    @override
    def dump(self, data: Mapping[str, object], path: Path) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message=r".*Please use DTensor instead.*"
            )

            def dump_error() -> TensorDumpError:
                return TensorDumpError(
                    path, f"The '{path}' tensor file cannot be dumped. See the nested exception for details.",  # fmt: skip
                )

            try:
                fp = self._file_system.open(path, mode=FileMode.WRITE)
            except OSError as ex:
                raise dump_error() from ex

            try:
                torch.save(data, fp)
            except (RuntimeError, OSError, PickleError) as ex:
                raise dump_error() from ex
            finally:
                fp.close()


@final
class SafetensorLoader(TensorLoader):
    """Loads the Hugging Face Safetensors file(s)."""

    _file_system: FileSystem

    def __init__(self, file_system: FileSystem) -> None:
        if not file_system.is_local:
            raise NotSupportedError("Safetensors supports only local file system.")

        self._file_system = file_system

    @override
    def load(
        self,
        path: Path,
        *,
        map_location: MapLocation = None,
        restrict: bool = True,
        mmap: bool = False,
    ) -> dict[str, object]:
        try:
            from safetensors import safe_open  # type: ignore[import-not-found]
        except ImportError:
            raise NotSupportedError(
                "Safetensors not found in your Python environment. Use `pip install safetensors`."
            )

        if map_location is not None:
            if not isinstance(map_location, (Device, str)):
                raise NotSupportedError(
                    "Safetensors supports only `torch.device` and `str` as `map_location`."
                )

        try:
            is_dir = self._file_system.is_dir(path)
        except OSError as ex:
            raise TensorLoadError(
                path, f"The '{path}' path cannot be accessed. See the nested exception for details."  # fmt: skip
            ) from ex

        if is_dir:
            try:
                files = list(self._file_system.glob(path, "*.safetensors"))
            except OSError as ex:
                raise TensorLoadError(
                    path, f"The '{path}' directory cannot be traversed. See the nested exception for details."  # fmt: skip
                ) from ex

            if not files:
                raise TensorLoadError(
                    path, f"No Safetensors file found under the '{path}' directory."  # fmt: skip
                )
        else:
            files = [path]

        tensors = {}

        for file in files:
            try:
                with safe_open(file, framework="pt", device=str(map_location)) as f:  # type: ignore[attr-defined]
                    for k in f.keys():
                        if k in tensors:
                            raise TensorLoadError(
                                path, f"The '{k}' key exists in more than one Safetensors file under the '{path}' directory."  # fmt: skip
                            )

                        tensors[k] = f.get_tensor(k)
            except FileNotFoundError:
                raise
            except (RuntimeError, OSError, PickleError) as ex:
                raise TensorLoadError(
                    file, f"The '{file}' tensor file cannot be loaded. See the nested exception for details."  # fmt: skip
                ) from ex

        return tensors


@final
class ShardedTensorLoader(TensorLoader):
    _inner_loader: TensorLoader
    _file_system: FileSystem
    _dim: int

    def __init__(
        self, inner_loader: TensorLoader, file_system: FileSystem, *, dim: int = 0
    ) -> None:
        self._inner_loader = inner_loader
        self._file_system = file_system
        self._dim = dim

    @override
    def load(
        self,
        path: Path,
        *,
        map_location: MapLocation = None,
        restrict: bool = True,
        mmap: bool = False,
    ) -> dict[str, object]:
        try:
            is_dir = self._file_system.is_dir(path)
        except OSError as ex:
            raise TensorLoadError(
                path, f"The '{path}' path cannot be accessed. See the nested exception for details."  # fmt: skip
            ) from ex

        if not is_dir:
            return self._inner_loader.load(
                path, map_location=map_location, restrict=restrict, mmap=mmap
            )

        # Determine the list of shard files.
        shard_files = []

        for shard_idx in count():
            shard_file = path.joinpath(f"shard.{shard_idx:02d}.pt")

            try:
                shard_file_exists = self._file_system.exists(shard_file)
            except OSError as ex:
                raise TensorLoadError(
                    path, f"The '{path}' path cannot be accessed. See the nested exception for details."  # fmt: skip
                ) from ex

            if not shard_file_exists:
                break

            shard_files.append(shard_file)

        if not shard_files:
            raise TensorLoadError(
                path, f"The '{path}' directory does not contain any sharded tensor files."  # fmt: skip
            )

        # Load the shards.
        shards: list[object] = []

        for idx, shard_file in enumerate(shard_files):
            shard = self._inner_loader.load(
                shard_file, map_location=map_location, restrict=restrict, mmap=mmap
            )

            shards.append(shard)

        output = self._unshard(path, shards, [])

        return cast(dict[str, object], output)

    def _unshard(
        self, path: Path, shards: list[object], item_path: list[str]
    ) -> object:
        first_shard = shards[0]

        if len(shards) == 1:
            return first_shard

        kls = type(first_shard)

        other_shards = shards[1:]

        idx = 1

        for other_shard in other_shards:
            if not isinstance(other_shard, kls):
                path = path.joinpath(f"shard_{idx:02d}.pt")

                item_pathname = ".".join(item_path)

                raise TensorLoadError(
                    path, f"The '{item_pathname}' item in the '{path}' file is expected to be of type `{kls}`, but is of type `{type(other_shard)}` instead."  # fmt: skip
                )

            idx += 1

        if issubclass(kls, dict):
            output = {}

            first_shard = cast(dict[str, object], first_shard)

            keys = list(first_shard.keys())

            for key in keys:
                item_path.append(key)

                item = first_shard.pop(key)

                inputs = [item]

                for other_shard in other_shards:
                    other_shard = cast(dict[str, object], other_shard)

                    try:
                        other_item = other_shard.pop(key)
                    except KeyError:
                        break

                    inputs.append(other_item)

                merged_item = self._unshard(path, inputs, item_path)

                item_path.pop()

                output[key] = merged_item

            return output

        if issubclass(kls, Tensor):
            tensors = cast(list[Tensor], shards)

            try:
                return torch.cat(tensors, dim=self._dim)
            except RuntimeError as ex:
                item_pathname = ".".join(item_path)

                raise TensorLoadError(
                    path, f"The shards of the '{item_pathname}' item under the '{path}' directory cannot be merged. See the nested exception for details."  # fmt: skip
                ) from ex

        # Everything except tensors are considered replicated.
        return first_shard


@final
class AutoTensorLoader(TensorLoader):
    _file_system: FileSystem
    _default_tensor_loader: TensorLoader

    def __init__(self, file_system: FileSystem) -> None:
        self._file_system = file_system

        self._default_tensor_loader = TorchTensorLoader(self._file_system)

    @override
    def load(
        self,
        path: Path,
        *,
        map_location: MapLocation = None,
        restrict: bool = True,
        mmap: bool = False,
    ) -> dict[str, object]:
        def has_file(extension: str) -> bool:
            try:
                next(iter(self._file_system.glob(path, f"*{extension}")))
            except OSError as ex:
                raise TensorLoadError(
                    path, f"The '{path}' directory cannot be traversed. See the nested exception for details."  # fmt: skip
                ) from ex
            except StopIteration:
                return False

            return True

        loader: TensorLoader

        try:
            is_dir = self._file_system.is_dir(path)
        except OSError as ex:
            raise TensorLoadError(
                path, f"The '{path}' path cannot be accessed. See the nested exception for details."  # fmt: skip
            ) from ex

        if is_dir:
            if has_file(".safetensors"):
                loader = SafetensorLoader(self._file_system)
            else:
                loader = ShardedTensorLoader(
                    self._default_tensor_loader, self._file_system, dim=0
                )
        elif path.suffix == ".safetensors":
            loader = SafetensorLoader(self._file_system)
        else:
            loader = self._default_tensor_loader

        return loader.load(
            path, map_location=map_location, restrict=restrict, mmap=mmap
        )


class TensorLoadError(Exception):
    path: Path

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)

        self.path = path


class TensorDumpError(Exception):
    path: Path

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)

        self.path = path
