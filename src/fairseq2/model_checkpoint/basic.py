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
from fairseq2.gang import Gangs
from fairseq2.io import TensorFileError, TensorLoader
from fairseq2.model_checkpoint.common import reshard_tensor
from fairseq2.model_checkpoint.loader import (
    ModelCheckpointError,
    ModelCheckpointLoader,
    StateDictConverter,
)
from fairseq2.sharder import ShardSpec


@final
class BasicModelCheckpointLoader(ModelCheckpointLoader):
    """Loads single-file PyTorch checkpoints (.pt, .pth, .bin)."""

    def __init__(self, file_system: FileSystem, tensor_loader: TensorLoader) -> None:
        self._file_system = file_system
        self._tensor_loader = tensor_loader

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
        try:
            checkpoint = self._tensor_loader.load(
                path, map_location=CPU, mmap=mmap, restrict=restrict
            )
        except TensorFileError as ex:
            msg = f"{path} is not a valid checkpoint file."

            raise ModelCheckpointError(path, msg) from ex

        if state_dict_converter is not None:
            checkpoint = state_dict_converter(checkpoint)

        source_shard_sizes = (1, 1)

        target_shard_sizes = (gangs.tp.size, gangs.sdp.size)
        target_shard_ranks = (gangs.tp.rank, gangs.sdp.rank)

        memo = set()

        for key, tensor in checkpoint.items():
            if not isinstance(tensor, Tensor):
                msg = f"{key} in {path} is not a `{Tensor}`."

                raise ModelCheckpointError(path, msg)

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
                shard_specs,
                shard_dims,
            )

            yield key, tensor

            del tensor

    @override
    def supports_path(self, path: Path) -> bool:
        if not path.suffix in (".pt", ".pth", ".bin"):
            return False

        return self._file_system.is_file(path)
