# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path
from pickle import PickleError
from typing import TypeAlias, final

import torch
from torch import Tensor
from typing_extensions import override

try:
    import safetensors  # type: ignore[import-not-found]
except ImportError:
    _has_safetensors = False
else:
    _has_safetensors = True

from fairseq2.device import Device
from fairseq2.error import NotSupportedError
from fairseq2.file_system import FileMode, FileSystem

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
            raise NotSupportedError("Safetensors supports only local file systems.")

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
        if not _has_safetensors:
            raise RuntimeError(
                "Safetensors is not found in your Python environment. Use `pip install safetensors`."
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
                with safetensors.safe_open(
                    file, framework="pt", device=str(map_location)
                ) as f:
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
            if not has_file(".safetensors"):
                raise TensorLoadError(
                    path, f"The '{path}' directory does not contain any supported tensor files."  # fmt: skip
                )

            loader = SafetensorLoader(self._file_system)
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
