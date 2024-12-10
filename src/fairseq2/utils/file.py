# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, TypeAlias
from warnings import catch_warnings

import torch
from torch import Tensor

from fairseq2.typing import Device

MapLocation: TypeAlias = (
    Callable[[Tensor, str], Tensor] | Device | str | dict[str, str] | None
)


class TensorLoader(Protocol):
    """Loads tensors from files."""

    def __call__(
        self,
        path: Path,
        *,
        map_location: MapLocation = None,
        restrict: bool = False,
    ) -> dict[str, Any]:
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

    def __call__(self, data: Mapping[str, Any], path: Path) -> None:
        """
        :param data:
            The dictionary containing tensors and other auxiliary data.
        :param path:
            The path to the file.
        """


def load_pt_tensors(
    path: Path,
    *,
    map_location: MapLocation = None,
    restrict: bool = False,
) -> dict[str, Any]:
    """Load the PyTorch tensor file stored under ``path``."""
    with catch_warnings():
        warnings.simplefilter("ignore")  # Suppress noisy FSDP warnings.

        data: dict[str, Any] = torch.load(
            str(path), map_location, weights_only=restrict  # type: ignore[arg-type]
        )

    return data


def dump_pt_tensors(data: Mapping[str, Any], path: Path) -> None:
    """Dump ``data`` to a PyTorch tensor file under ``path``."""
    with catch_warnings():
        warnings.simplefilter("ignore")  # Suppress noisy FSDP warnings.

        torch.save(data, path)


def load_safetensors(
    path: Path,
    *,
    map_location: MapLocation = None,
    restrict: bool = False,
) -> dict[str, Any]:
    """Load the Hugging Face Safetensors file(s) stored under ``path``."""
    try:
        from safetensors import safe_open  # type: ignore[import-not-found]
    except ImportError:
        raise RuntimeError(
            "Safetensors not found in your Python environment. Use `pip install safetensors`."
        )

    if map_location is not None:
        if not isinstance(map_location, (Device, str)):
            raise RuntimeError(
                "Safetensors only supports `torch.device` and `str` for the `map_location` parameter."
            )

    if path.is_dir():
        files = _get_files(path, ".safetensors")
        if not files:
            raise RuntimeError(
                f"No Safetensors file found under the directory '{path}'."
            )
    else:
        files = [path]

    tensors = {}

    for file in files:
        with safe_open(file, framework="pt", device=str(map_location)) as f:  # type: ignore[attr-defined]
            for k in f.keys():
                if k in tensors:
                    raise RuntimeError(
                        f"The '{k}' key exists in more than one Safetensors file under the directory '{path}'."
                    )

                tensors[k] = f.get_tensor(k)

    return tensors


def load_tensors(
    path: Path,
    *,
    map_location: MapLocation = None,
    restrict: bool = False,
) -> dict[str, Any]:
    """Load the tensors stored under ``path``."""
    loader = load_pt_tensors

    if path.is_dir():
        if not _has_files(path, ".safetensors"):
            raise RuntimeError(f"'{path}' is a directory with no known tensor files.")

        loader = load_safetensors
    elif path.suffix == ".safetensors":
        loader = load_safetensors

    return loader(path, map_location=map_location, restrict=restrict)


def _get_files(path: Path, extension: str) -> list[Path]:
    return list(path.glob("*" + extension))


def _has_files(path: Path, extension: str) -> bool:
    try:
        next(iter(path.glob("*" + extension)))
    except StopIteration:
        return False

    return True
