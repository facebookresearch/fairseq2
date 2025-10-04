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
    load model state by yielding parameters lazily rather than loading
    everything into memory at once.
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

        Yields tensors one at a time to minimize memory usage. Supports tensor
        resharding and optional state dictionary conversion.

        If ``mmap`` is ``True``, uses memory mapping when possible to reduce
        memory footprint.

        If ``restrict`` is ``True``, restricts pickle (if used) to load only
        tensors and other types that can be safely serialized and deserialized.

        If ``state_dict_converter`` is specified, it will be used to transform
        the state dictionaries in the checkpoint. Typically used to convert from
        one format such as Hugging Face Transformers to fairseq2.

        ``shard_dims`` specifies the sharding dimension for each parameter as
        returned by :func:`~fairseq2.nn.get_sharding_dims`. Along with ``gangs``,
        this enables on-the-fly parameter sharding/resharding during checkpoint
        loading. If ``None``, no sharding/resharding will be performed and the
        full tensor will be loaded.

        ``shard_specs`` is deprecated and will be removed in a future release;
        please use ``shard_dims`` instead.

        Yields pairs of ``(parameter name, parameter)`` for each parameter in
        the checkpoint.

        :raises ModelCheckpointError: If the checkpoint is not valid.
        """

    @abstractmethod
    def supports_path(self, path: Path) -> bool:
        """Checks if this loader can handle the specified checkpoint path."""


class ModelCheckpointError(Exception):
    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)

        self.path = path
