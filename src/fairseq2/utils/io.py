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
    import safetensors.torch  # type: ignore[import-not-found]
except ImportError:
    _has_safetensors = False
else:
    _has_safetensors = True

from fairseq2.device import CPU, Device
from fairseq2.error import NotSupportedError
from fairseq2.file_system import FileMode, FileSystem

MapLocation: TypeAlias = (
    Callable[[Tensor, str], Tensor] | Device | str | dict[str, str] | None
)


class TensorLoader(ABC):
    """Loads tensors from PyTorch binary files."""

    @abstractmethod
    def load(
        self,
        file: Path,
        *,
        map_location: MapLocation = None,
        mmap: bool = False,
        restrict: bool = True,
    ) -> dict[str, object]:
        """
        :param file: The path to the file.
        :param map_location: Same as the ``map_location`` parameter of
            :meth:`torch.load`.
        """


class TensorDumper(ABC):
    """Dumps tensors to PyTorch binary files."""

    @abstractmethod
    def dump(self, data: Mapping[str, object], file: Path) -> None:
        """
        :param data: The dictionary containing tensors and other auxiliary data.
        :param file: The path to the file.
        """


@final
class TorchTensorLoader(TensorLoader):
    _file_system: FileSystem

    def __init__(self, file_system: FileSystem) -> None:
        self._file_system = file_system

    @override
    def load(
        self,
        file: Path,
        *,
        map_location: MapLocation = None,
        mmap: bool = False,
        restrict: bool = True,
    ) -> dict[str, object]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message=r".*You are using `torch\.load` with `weights_only=False`.*"  # fmt: skip
            )

            def load_error() -> TensorLoadError:
                return TensorLoadError(
                    file, f"The '{file}' tensor file cannot be loaded. See the nested exception for details."  # fmt: skip
                )

            try:
                data: dict[str, object] = torch.load(
                    file, map_location, weights_only=restrict, mmap=mmap  # type: ignore[arg-type]
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
    def dump(self, data: Mapping[str, object], file: Path) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message=r".*Please use DTensor instead.*"
            )

            def dump_error() -> TensorDumpError:
                return TensorDumpError(
                    file, f"The '{file}' tensor file cannot be dumped. See the nested exception for details.",  # fmt: skip
                )

            try:
                fp = self._file_system.open(file, mode=FileMode.WRITE)
            except OSError as ex:
                raise dump_error() from ex

            try:
                torch.save(data, fp)
            except (RuntimeError, OSError, PickleError) as ex:
                raise dump_error() from ex
            finally:
                fp.close()


class SafetensorsLoader(ABC):
    """Loads Safetensors files."""

    @abstractmethod
    def load(
        self, file: Path, *, device: Device | None = None, mmap: bool = False
    ) -> dict[str, object]: ...


@final
class HuggingFaceSafetensorsLoader(SafetensorsLoader):
    _file_system: FileSystem

    def __init__(self, file_system: FileSystem) -> None:
        if not file_system.is_local:
            raise NotSupportedError("Safetensors supports only local file systems.")

        self._file_system = file_system

    @override
    def load(
        self, file: Path, *, device: Device | None = None, mmap: bool = False
    ) -> dict[str, object]:
        if not _has_safetensors:
            raise RuntimeError(
                "Safetensors is not found in your Python environment. Use `pip install safetensors`."
            )

        if device is None:
            device = CPU

        data = {}

        try:
            if mmap:
                with safetensors.safe_open(
                    file, framework="pt", device=str(device)
                ) as f:
                    for key in f.keys():
                        data[key] = f.get_tensor(key)
            else:
                with open(file, "rb") as f:
                    bits = f.read()

                tensors = safetensors.torch.load(bits)

                for key, tensor in tensors.items():
                    data[key] = tensor.to(device)

        except FileNotFoundError:
            raise
        except (RuntimeError, OSError, PickleError) as ex:
            raise TensorLoadError(
                file, f"The '{file}' tensor file cannot be loaded. See the nested exception for details."  # fmt: skip
            ) from ex

        return data


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
