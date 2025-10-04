# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import final

from torch import Tensor
from typing_extensions import override

from fairseq2.file_system import FileSystem, raise_if_not_exists
from fairseq2.gang import Gangs
from fairseq2.model_checkpoint.loader import (
    ModelCheckpointError,
    ModelCheckpointLoader,
    StateDictConverter,
)
from fairseq2.sharder import ShardSpec


@final
class DelegatingModelCheckpointLoader(ModelCheckpointLoader):
    """
    Delegates loading to format-specific checkpoint loaders.

    This loader maintains a collection of specialized loaders and automatically
    selects the appropriate one based on the checkpoint file format. It provides
    a unified interface for loading various checkpoint formats without requiring
    the caller to handle format-specific logic.

    The loader iterates through its registered loaders in order and uses the
    first one that reports it can handle the given path via
    :meth:`ModelCheckpointLoader.supports_path()`.
    """

    def __init__(
        self, loaders: Sequence[ModelCheckpointLoader], file_system: FileSystem
    ) -> None:
        self._loaders = loaders
        self._file_system = file_system

    @override
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
        raise_if_not_exists(self._file_system, path)

        for loader in self._loaders:
            if loader.supports_path(path):
                return loader.lazy_load(
                    path,
                    gangs,
                    mmap=mmap,
                    restrict=restrict,
                    state_dict_converter=state_dict_converter,
                    shard_specs=shard_specs,
                    shard_dims=shard_dims,
                )

        msg = f"{path} does not point to any known checkpoints."

        raise ModelCheckpointError(path, msg)

    @override
    def supports_path(self, path: Path) -> bool:
        for loader in self._loaders:
            if loader.supports_path(path):
                return True

        return False
