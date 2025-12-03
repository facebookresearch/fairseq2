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

from fairseq2.file_system import FileSystem, _raise_file_not_found_error
from fairseq2.model_checkpoint.loader import (
    BadModelCheckpointError,
    ModelCheckpointLoader,
    ModelCheckpointLoadOptions,
)


@final
class _DelegatingModelCheckpointLoader(ModelCheckpointLoader):
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
        shard_dims: Mapping[str, int],
        options: ModelCheckpointLoadOptions | None = None,
    ) -> Iterator[tuple[str, Tensor]]:
        if options is None:
            options = ModelCheckpointLoadOptions()

        try:
            path_exists = self._file_system.exists(path)
        except OSError as ex:
            raise OSError(
                f"an I/O error occurred while accessing path '{path}'"
            ) from ex

        if not path_exists:
            _raise_file_not_found_error(path)

        for loader in self._loaders:
            if loader.supports_path(path):
                return loader.lazy_load(path, shard_dims, options)

        raise BadModelCheckpointError(
            f"checkpoint '{path}' does not have a known format"
        )

    @override
    def supports_path(self, path: Path) -> bool:
        for loader in self._loaders:
            if loader.supports_path(path):
                return True

        return False
