# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Protocol

from torch import Tensor

from fairseq2.gang import Gangs
from fairseq2.sharder import ShardSpec


class StateDictConverter(Protocol):
    def __call__(self, state_dict: dict[str, object]) -> dict[str, object]: ...


class ModelCheckpointLoader(ABC):
    """
    Represents the abstract base class for model checkpoint loaders.

    This class defines the interface for checkpoint loaders that can efficiently
    load model state by yielding parameters lazily rather than loading everything
    into memory at once.
    """

    @abstractmethod
    def lazy_load(
        self,
        path: Path,
        gangs: Gangs,
        *,
        mmap: bool = False,
        restrict: bool = True,
        state_dict_converter: StateDictConverter | None = None,
        shard_specs: Mapping[str, ShardSpec] | None = None,
        shard_dims: Mapping[str, int] | None = None,
    ) -> Iterator[tuple[str, Tensor]]:
        """
        Lazily loads parameters from the specified checkpoint path.

        Yields tensors one at a time to minimize memory usage if the underlying
        format allows it. Supports tensor resharding and optional state dictionary
        conversion.

        ``gangs`` is used to determine the distributed target configuration and
        shard yielded parameters accordingly.

        If ``mmap`` is ``True``, the checkpoint will be memory-mapped. This can
        reduce memory usage but may cause slower load times on some systems.

        If ``restrict`` is ``True``, pickle (if used) will be restricted to load
        only tensors and types that can be safely serialized and deserialized.

        If ``state_dict_converter`` is provided, it will be used to transform
        the (sharded) state dictionaries in the checkpoint. Typically used to
        convert from one format such as Hugging Face Transformers to fairseq2.

        If ``shard_dims`` is provided, it specifies the sharding dimension of
        each parameter as returned by :func:`~fairseq2.nn.get_sharding_dims`.
        Along with ``gangs``, they enable on-the-fly parameter resharding during
        checkpoint loading. If ``None``, no resharding will be performed and
        full parameters will be loaded.

        ``shard_specs`` is deprecated and will be removed in v0.12; please use
        ``shard_dims`` instead.

        Yields pairs of ``(parameter name, parameter)`` for each parameter in
        the checkpoint.

        :raises ModelCheckpointError: If the checkpoint is not valid.
        """

    @abstractmethod
    def supports_path(self, path: Path) -> bool:
        """Checks if this loader can handle the specified checkpoint path."""


class ModelCheckpointError(Exception):
    """Raised when a model checkpoint is not valid."""

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)

        self.path = path
