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

import safetensors
import safetensors.torch
import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.device import CPU, Device
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


class TensorFileError(Exception):
    def __init__(self, file: Path, message: str) -> None:
        super().__init__(message)

        self.file = file


class TensorDumper(ABC):
    """Dumps tensors to PyTorch binary files."""

    @abstractmethod
    def dump(
        self, data: Mapping[str, object], file: Path, *, pickle_protocol: int = 2
    ) -> None:
        """
        :param data: The dictionary containing tensors and other auxiliary data.
        :param file: The path to the file.
        """


class TensorDataNotValidError(Exception):
    pass


@final
class TorchTensorLoader(TensorLoader):
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

            try:
                data: dict[str, object] = torch.load(
                    file, map_location, weights_only=restrict, mmap=mmap  # type: ignore[arg-type]
                )
            except (RuntimeError, PickleError, EOFError) as ex:
                msg = f"{file} is not a valid PyTorch tensor file."

                raise TensorFileError(file, msg) from ex

        return data


@final
class TorchTensorDumper(TensorDumper):
    def __init__(self, file_system: FileSystem) -> None:
        self._file_system = file_system

    @override
    def dump(
        self, data: Mapping[str, object], file: Path, *, pickle_protocol: int = 2
    ) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message=r".*Please use DTensor instead.*"
            )

            fp = self._file_system.open(file, mode=FileMode.WRITE)

            try:
                torch.save(data, fp, pickle_protocol=pickle_protocol)
            except (RuntimeError, PickleError) as ex:
                raise TensorDataNotValidError(
                    "`data` is not a pickleable object."
                ) from ex
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
    def __init__(self, file_system: FileSystem) -> None:
        self._file_system = file_system

    @override
    def load(
        self, file: Path, *, device: Device | None = None, mmap: bool = False
    ) -> dict[str, object]:
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
                fp = self._file_system.open(file, mode=FileMode.READ)

                with fp:
                    bits = fp.read()

                tensors = safetensors.torch.load(bits)

                for key, tensor in tensors.items():
                    data[key] = tensor.to(device)
        except (RuntimeError, PickleError, EOFError) as ex:
            msg = f"{file} is not a valid Safetensors file."

            raise TensorFileError(file, msg) from ex

        return data
