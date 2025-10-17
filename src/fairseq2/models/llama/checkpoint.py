# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import count
from pathlib import Path
from typing import final

from torch import Tensor
from typing_extensions import override

from fairseq2.device import CPU
from fairseq2.file_system import FileSystem, raise_if_not_exists
from fairseq2.gang import Gangs
from fairseq2.io import TensorFileError, TensorLoader
from fairseq2.model_checkpoint import (
    ModelCheckpointError,
    ModelCheckpointLoader,
    StateDictConverter,
    reshard_tensor,
)
from fairseq2.sharder import ShardSpec


@final
class LLaMACheckpointLoader(ModelCheckpointLoader):
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
        raise_if_not_exists(self._file_system, path)

        is_dir = self._file_system.is_dir(path)
        if not is_dir:
            msg = f"{path} does not point to a LLaMA checkpoint."

            raise ModelCheckpointError(path, msg)

        tp_files = self._get_checkpoint_files(path)

        tp_size = len(tp_files)

        source_shard_sizes = (tp_size, 1)

        target_shard_sizes = (gangs.tp.size, gangs.sdp.size)
        target_shard_ranks = (gangs.tp.rank, gangs.sdp.rank)

        # If the source and target tensor parallel sizes match, avoid loading
        # redundant checkpoint files.
        if gangs.tp.size == tp_size:
            tp_files = [tp_files[gangs.tp.rank]]

            source_shard_sizes = (1, 1)

            target_shard_sizes = (1, gangs.sdp.size)
            target_shard_ranks = (0, gangs.sdp.rank)

        # Load the checkpoint files.
        tp_shards = []

        for tp_file in tp_files:
            try:
                tp_shard = self._tensor_loader.load(
                    tp_file, map_location=CPU, mmap=mmap, restrict=restrict
                )
            except TensorFileError as ex:
                msg = f"{tp_file} is not a valid checkpoint file."

                raise ModelCheckpointError(path, msg) from ex

            if state_dict_converter is not None:
                tp_shard = state_dict_converter(tp_shard)

            tp_shards.append(tp_shard)

        memo = set()

        # Assume that the very first tensor parallel shard contains all the
        # checkpoint keys.
        keys = list(tp_shards[0].keys())

        for key in keys:
            splits = []

            for tp_shard in tp_shards:
                try:
                    tp_split = tp_shard.pop(key)
                except KeyError:
                    break  # data parallel sharding can be uneven.

                if not isinstance(tp_split, Tensor):
                    msg = f"{key} in {path} is not a `{Tensor}`."

                    raise ModelCheckpointError(path, msg)

                splits.append([tp_split])

            split_0 = splits[0][0]

            if split_0 in memo:  # Yield shared tensors only once.
                continue

            memo.add(split_0)

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

    def _get_checkpoint_files(self, path: Path) -> list[Path]:
        tp_files = []

        for tp_idx in count():
            tp_file = path.joinpath(f"consolidated.{tp_idx:02d}.pth")

            is_file = self._file_system.is_file(tp_file)
            if not is_file:
                break

            tp_files.append(tp_file)

        if not tp_files:
            msg = f"{path} does not contain any tensor files."

            raise ModelCheckpointError(path, msg)

        return tp_files

    @override
    def supports_path(self, path: Path) -> bool:
        is_dir = self._file_system.is_dir(path)
        if not is_dir:
            return False

        file = path.joinpath("consolidated.00.pth")

        return self._file_system.is_file(file)
