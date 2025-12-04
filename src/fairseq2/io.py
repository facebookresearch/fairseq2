# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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


@dataclass(kw_only=True)
class TensorFileLoadOptions:
    map_location: MapLocation = None
    """See the ``map_location`` parameter of :func:`torch.load`."""

    mmap: bool = False
    """
    Indicates whether the tensor file should be memory-mapped rather than
    loading it into memory.
    """

    restrict: bool = True
    """
    Indicates whether unpickler should be restricted to loading only tensors,
    primitive types, dictionaries.
    """


class TensorFileLoader(ABC):
    """Loads tensors from PyTorch tensor files."""

    @abstractmethod
    def load(
        self, file: Path, options: TensorFileLoadOptions | None = None
    ) -> dict[str, object]:
        """
        :raises CorruptFileError: Specified file is erroneous and cannot be
            loaded as a PyTorch tensor file.

        :raises OSError: A system error occurred.
        """


# Legacy name
TensorLoader: TypeAlias = TensorFileLoader


class CorruptFileError(Exception):
    def __init__(self, file: Path) -> None:
        super().__init__(f"{file} is erroneous and cannot be loaded.")

        self.file = file


@dataclass(kw_only=True)
class TensorFileDumpOptions:
    pickle_protocol: int = 2


class TensorFileDumper(ABC):
    """Dumps tensors to PyTorch tensor files."""

    @abstractmethod
    def dump(
        self,
        data: Mapping[str, object],
        file: Path,
        options: TensorFileDumpOptions | None = None,
    ) -> None:
        """
        :raises DataNotPicklableError: Specified data is not picklable.

        :raises OSError: A system error occurred.
        """


# Legacy name
TensorDumper: TypeAlias = TensorFileDumper


class DataNotPicklableError(Exception):
    def __init__(self) -> None:
        super().__init__("Data is not picklable.")


@final
class _TorchTensorFileLoader(TensorFileLoader):
    def __init__(self, file_system: FileSystem) -> None:
        self._file_system = file_system

    @override
    def load(
        self, file: Path, options: TensorFileLoadOptions | None = None
    ) -> dict[str, object]:
        if options is None:
            options = TensorFileLoadOptions()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message=r".*You are using `torch\.load` with `weights_only=False`.*"  # fmt: skip
            )

            try:
                data: dict[str, object] = torch.load(
                    file,
                    options.map_location,  # type: ignore[arg-type]
                    weights_only=options.restrict,
                    mmap=options.mmap,
                )
            except (RuntimeError, PickleError, EOFError) as ex:
                raise CorruptFileError(file) from ex

        return data


@final
class _TorchTensorFileDumper(TensorFileDumper):
    def __init__(self, file_system: FileSystem) -> None:
        self._file_system = file_system

    @override
    def dump(
        self,
        data: Mapping[str, object],
        file: Path,
        options: TensorFileDumpOptions | None = None,
    ) -> None:
        if options is None:
            options = TensorFileDumpOptions()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message=r".*Please use DTensor instead.*"
            )

            fp = self._file_system.open(file, mode=FileMode.WRITE)

            try:
                torch.save(data, fp, pickle_protocol=options.pickle_protocol)
            except (RuntimeError, PickleError) as ex:
                raise DataNotPicklableError() from ex
            finally:
                fp.close()


@dataclass
class SafetensorsLoadOptions:
    device: Device | None = None
    """
    Device on which to load the tensors. If ``None``, tensors will be loaded
    on CPU.
    """

    mmap: bool = False
    """
    Indicates whether the tensor file should be memory-mapped rather than
    loading it into memory.
    """


class SafetensorsLoader(ABC):
    """Loads Hugging Face Safetensors files."""

    @abstractmethod
    def load(
        self, file: Path, options: SafetensorsLoadOptions | None = None
    ) -> dict[str, object]:
        """
        :raises CorruptFileError: Specified file is erroneous and cannot be
            loaded as a Safetensors file.

        :raises OSError: A system error occurred.
        """


@final
class _HuggingFaceSafetensorsLoader(SafetensorsLoader):
    def __init__(self, file_system: FileSystem) -> None:
        self._file_system = file_system

    @override
    def load(
        self, file: Path, options: SafetensorsLoadOptions | None = None
    ) -> dict[str, object]:
        if options is None:
            options = SafetensorsLoadOptions()

        device = options.device
        if device is None:
            device = CPU

        data = {}

        try:
            if options.mmap:
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
            raise CorruptFileError(file) from ex

        return data
