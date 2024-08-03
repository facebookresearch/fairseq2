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


def load_tensors(
    path: Path,
    *,
    map_location: MapLocation = None,
    restrict: bool = False,
) -> dict[str, Any]:
    """Load the PyTorch tensor file stored under ``path``."""
    with catch_warnings():
        warnings.simplefilter("ignore")  # Suppress the deprecation warning.

        data: dict[str, Any] = torch.load(
            str(path), map_location, weights_only=restrict  # type: ignore[arg-type]
        )

    return data


def dump_tensors(data: Mapping[str, Any], path: Path) -> None:
    """Dump ``data`` to a PyTorch tensor file under ``path``."""
    torch.save(data, path)
