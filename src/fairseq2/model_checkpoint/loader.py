# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from torch import Tensor

from fairseq2.gang import Gangs
from fairseq2.runtime.dependency import get_dependency_resolver


def get_model_checkpoint_loader() -> ModelCheckpointLoader:
    resolver = get_dependency_resolver()

    return resolver.resolve(ModelCheckpointLoader)


class StateDictConverter(Protocol):
    def __call__(self, state_dict: dict[str, object]) -> dict[str, object]: ...


@dataclass(kw_only=True)
class ModelCheckpointLoadOptions:
    gangs: Gangs | None = None
    """
    Used to determine the distributed target configuration and shard yielded
    parameters accordingly. If ``None``, the gangs returned from
    :func:`get_current_gangs` will be used.
    """

    mmap: bool = False
    """
    Indicates whether the checkpoint will be memory-mapped. This can reduce
    memory usage but may cause slower load times on some systems.
    """

    restrict: bool = True
    """
    Indicates whether unpickler (if used) will be restricted to load only
    tensors and types that can be safely serialized and deserialized.
    """

    state_dict_converter: StateDictConverter | None = None
    """
    If provided, used to transform the (sharded) state dictionaries in the
    checkpoint from one format, such as Hugging Face Transformers, to fairseq2.
    """


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
        shard_dims: Mapping[str, int],
        options: ModelCheckpointLoadOptions | None = None,
    ) -> Iterator[tuple[str, Tensor]]:
        """
        Lazily loads parameters from the specified checkpoint path.

        Yields tensors one at a time to minimize memory usage if the underlying
        format allows it. Supports tensor resharding and optional state
        dictionary conversion.

        If ``shard_dims`` is provided, it specifies the sharding dimension of
        each parameter as returned by :func:`~fairseq2.nn.get_sharding_dims`.
        Along with ``gangs``, they enable on-the-fly parameter resharding during
        checkpoint loading.

        Yields pairs of ``(parameter name, parameter)`` for each parameter in
        the checkpoint.

        :raises CorruptModelCheckpointError: Checkpoint is erroneous and cannot
            be loaded.

        :raises OSError: A system error occurred.
        """

    @abstractmethod
    def supports_path(self, path: Path) -> bool:
        """
        Checks if this loader can handle the specified checkpoint path.

        :raises OSError: A system error occurred.
        """


class CorruptModelCheckpointError(Exception):
    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)

        self.path = path
