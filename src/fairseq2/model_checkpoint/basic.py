# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import final

from torch import Tensor
from typing_extensions import override

from fairseq2.device import CPU
from fairseq2.file_system import FileSystem
from fairseq2.gang import GangContext
from fairseq2.io import CorruptFileError, TensorFileLoader, TensorFileLoadOptions
from fairseq2.model_checkpoint.common import reshard_tensor
from fairseq2.model_checkpoint.loader import (
    CorruptModelCheckpointError,
    ModelCheckpointLoader,
    ModelCheckpointLoadOptions,
)


@final
class _BasicModelCheckpointLoader(ModelCheckpointLoader):
    """Loads single-file PyTorch checkpoints (.pt, .pth, .bin)."""

    def __init__(
        self,
        file_system: FileSystem,
        tensor_file_loader: TensorFileLoader,
        gang_context: GangContext,
    ) -> None:
        self._file_system = file_system
        self._tensor_file_loader = tensor_file_loader
        self._gang_context = gang_context

    @override
    def lazy_load(
        self,
        path: Path,
        shard_dims: Mapping[str, int],
        options: ModelCheckpointLoadOptions | None = None,
    ) -> Iterator[tuple[str, Tensor]]:
        if options is None:
            options = ModelCheckpointLoadOptions()

        load_options = TensorFileLoadOptions(
            map_location=CPU, mmap=options.mmap, restrict=options.restrict
        )

        try:
            checkpoint = self._tensor_file_loader.load(path, load_options)
        except CorruptFileError as ex:
            message = f"{path} cannot be loaded as a PyTorch checkpoint file."

            raise CorruptModelCheckpointError(path, message) from ex

        if options.state_dict_converter is not None:
            checkpoint = options.state_dict_converter(checkpoint)

        gangs = options.gangs or self._gang_context.get_current_gangs()

        source_shard_sizes = (1, 1)

        target_shard_sizes = (gangs.tp.size, gangs.sdp.size)
        target_shard_ranks = (gangs.tp.rank, gangs.sdp.rank)

        memo = set()

        for key, tensor in checkpoint.items():
            if not isinstance(tensor, Tensor):
                message = f"{key} in {path} is not a `{Tensor}`."

                raise CorruptModelCheckpointError(path, message)

            if tensor in memo:  # Yield shared tensors only once.
                continue

            memo.add(tensor)

            splits = [[tensor]]  # tp, dp

            tensor = reshard_tensor(
                key,
                splits,
                source_shard_sizes,
                target_shard_sizes,
                target_shard_ranks,
                shard_dims,
            )

            yield key, tensor

            del tensor

    @override
    def supports_path(self, path: Path) -> bool:
        if not path.suffix in (".pt", ".pth", ".bin"):
            return False

        return self._file_system.is_file(path)
