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
    Indicates whether the tensor file should be memory-mapped rather than loaded
    into memory.
    """

    restrict: bool = True
    """
    Indicates whether unpickler should be restricted to loading only tensors,
    primitive types, lists, and dictionaries.
    """


class TensorFileLoader(ABC):
    """Loads tensors from PyTorch tensor files."""

    @abstractmethod
    def load(
        self, file: Path, options: TensorFileLoadOptions | None = None
    ) -> dict[str, object]:
        """
        :raises PickleError: File is not a valid PyTorch tensor file.

        :raises EOFError: File is not a valid PyTorch tensor file.

        :raises FileNotFoundError: File is not found.

        :raises OSError: An I/O error occurred.
        """


# Legacy name
TensorLoader: TypeAlias = TensorFileLoader


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

            data: dict[str, object] = torch.load(
                file,
                options.map_location,  # type: ignore[arg-type]
                weights_only=options.restrict,
                mmap=options.mmap,
            )

        return data


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
        :raises PickleError: Specified data is not picklable.

        :raises OSError: An I/O error occurred.
        """


# Legacy name
TensorDumper: TypeAlias = TensorFileDumper


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
            with fp:
                torch.save(data, fp, pickle_protocol=options.pickle_protocol)


@dataclass
class SafetensorsLoadOptions:
    device: Device | None = None
    """
    Device on which to load the tensors. If ``None``, tensors will be loaded
    on CPU.
    """

    mmap: bool = False
    """
    Indicates whether the tensor file should be memory-mapped rather than loaded
    into memory.
    """


class SafetensorsLoader(ABC):
    """Loads Hugging Face Safetensors files."""

    @abstractmethod
    def load(
        self, file: Path, options: SafetensorsLoadOptions | None = None
    ) -> dict[str, object]:
        """
        :raises SafetensorError: File is not a valid Safetensors file.

        :raises EOFError: File is not a valid Safetensors file.

        :raises FileNotFoundError: File is not found.

        :raises OSError: An I/O error occurred.
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

        if options.mmap:
            f = safetensors.safe_open(file, framework="pt", device=str(device))
            with f:
                for key in f.keys():
                    data[key] = f.get_tensor(key)
        else:
            fp = self._file_system.open(file, mode=FileMode.READ)
            with fp:
                bits = fp.read()

            tensors = safetensors.torch.load(bits)

            for key, tensor in tensors.items():
                data[key] = tensor.to(device)

        return data
