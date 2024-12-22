# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from pickle import PickleError
from typing import Protocol, TypeAlias, final
from warnings import catch_warnings

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.error import NotSupportedError
from fairseq2.typing import Device


class FileSystem(ABC):
    @abstractmethod
    def is_file(self, path: Path) -> bool:
        ...

    @abstractmethod
    def make_directory(self, path: Path) -> None:
        ...

    @abstractmethod
    def walk_directory(
        self, path: Path, *, on_error: Callable[[OSError], None] | None
    ) -> Iterable[tuple[str, Sequence[str]]]:
        ...

    @abstractmethod
    def resolve(self, path: Path) -> Path:
        ...


@final
class StandardFileSystem(FileSystem):
    @override
    def is_file(self, path: Path) -> bool:
        return path.is_file()

    @override
    def make_directory(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    @override
    def walk_directory(
        self, path: Path, *, on_error: Callable[[OSError], None] | None
    ) -> Iterable[tuple[str, Sequence[str]]]:
        for dir_pathname, _, filenames in os.walk(path, onerror=on_error):
            yield dir_pathname, filenames

    @override
    def resolve(self, path: Path) -> Path:
        return path.expanduser().resolve()


MapLocation: TypeAlias = (
    Callable[[Tensor, str], Tensor] | Device | str | dict[str, str] | None
)


class TensorLoader(Protocol):
    """Loads tensors from files."""

    def __call__(
        self, path: Path, *, map_location: MapLocation = None, restrict: bool = False
    ) -> dict[str, object]:
        """
        :param path:
            The path to the file.
        :param map_location:
            Same as the ``map_location`` parameter of :meth:`torch.load`.
        :param restrict:
            If ``True``, restricts the Python unpickler to load only tensors,
            primitive types, and dictionaries.
        """


class TensorDumper(Protocol):
    """Dumps tensors to files."""

    def __call__(self, data: Mapping[str, object], path: Path) -> None:
        """
        :param data:
            The dictionary containing tensors and other auxiliary data.
        :param path:
            The path to the file.
        """


def load_torch_tensors(
    path: Path, *, map_location: MapLocation = None, restrict: bool = False
) -> dict[str, object]:
    """Load the PyTorch tensor file stored under ``path``."""
    with catch_warnings():
        warnings.simplefilter("ignore")  # Suppress noisy FSDP warnings.

        try:
            data: dict[str, object] = torch.load(
                str(path), map_location, weights_only=restrict  # type: ignore[arg-type]
            )
        except FileNotFoundError:
            raise
        except (RuntimeError, OSError, PickleError) as ex:
            raise TensorLoadError(
                f"The '{path}' tensor file cannot be loaded. See the nested exception for details."
            ) from ex

    return data


def dump_torch_tensors(data: Mapping[str, object], path: Path) -> None:
    """Dump ``data`` to a PyTorch tensor file under ``path``."""
    with catch_warnings():
        warnings.simplefilter("ignore")  # Suppress noisy FSDP warnings.

        try:
            torch.save(data, path)
        except (RuntimeError, OSError, PickleError) as ex:
            raise TensorDumpError(
                f"The '{path}' tensor file cannot be dumped. See the nested exception for details.",
            ) from ex


def load_safetensors(
    path: Path, *, map_location: MapLocation = None, restrict: bool = False
) -> dict[str, object]:
    """Load the Hugging Face Safetensors file(s) stored under ``path``."""
    try:
        from safetensors import safe_open  # type: ignore[import-not-found]
    except ImportError:
        raise NotSupportedError(
            "Safetensors not found in your Python environment. Use `pip install safetensors`."
        )

    if map_location is not None:
        if not isinstance(map_location, (Device, str)):
            raise NotSupportedError(
                "Safetensors only supports `torch.device` and `str` for the `map_location` parameter."
            )

    if path.is_dir():
        files = list(path.glob("*.safetensors"))
        if not files:
            raise TensorLoadError(
                f"No Safetensors file found under the '{path}' directory."
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
                            f"The '{k}' key exists in more than one Safetensors file under the '{path}' directory."
                        )

                    tensors[k] = f.get_tensor(k)
        except FileNotFoundError:
            raise
        except (RuntimeError, OSError, PickleError) as ex:
            raise TensorLoadError(
                f"The '{file}' tensor file cannot be loaded. See the nested exception for details."
            ) from ex

    return tensors


def load_tensors(
    path: Path, *, map_location: MapLocation = None, restrict: bool = False
) -> dict[str, object]:
    """Load the tensors stored under ``path``."""

    def has_files(path: Path, extension: str) -> bool:
        try:
            next(iter(path.glob("*" + extension)))
        except StopIteration:
            return False

        return True

    if path.is_dir():
        if not has_files(path, ".safetensors"):
            raise TensorLoadError(
                f"The '{path}' directory does not contain any supported tensor files."
            )

        loader = load_safetensors
    elif path.suffix == ".safetensors":
        loader = load_safetensors
    else:
        loader = load_torch_tensors

    return loader(path, map_location=map_location, restrict=restrict)


class TensorLoadError(Exception):
    pass


class TensorDumpError(Exception):
    pass
