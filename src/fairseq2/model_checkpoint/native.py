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
from fairseq2.model_checkpoint.common import reshard_tensor
from fairseq2.model_checkpoint.loader import (
    ModelCheckpointError,
    ModelCheckpointLoader,
    StateDictConverter,
)
from fairseq2.sharder import ShardSpec


@final
class NativeModelCheckpointLoader(ModelCheckpointLoader):
    """
    Loads native fairseq2 checkpoints.

    The native fairseq2 format is optimized for efficient storage and loading of
    model checkpoints in distributed configurations.
    """

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
            msg = f"{path} does not point to a fairseq2 checkpoint."

            raise ModelCheckpointError(path, msg)

        pp_files = self._get_checkpoint_files(path)

        pp_size = len(pp_files)
        tp_size = len(pp_files[0])
        dp_size = len(pp_files[0][0])

        source_shard_sizes = (tp_size, dp_size)

        target_shard_sizes = (gangs.tp.size, gangs.sdp.size)
        target_shard_ranks = (gangs.tp.rank, gangs.sdp.rank)

        # If the source and target pipeline and tensor parallel sizes match,
        # avoid loading redundant checkpoint files.
        if gangs.pp.size == pp_size:
            if gangs.tp.size == tp_size:
                pp_files = [[pp_files[gangs.pp.rank][gangs.tp.rank]]]

                source_shard_sizes = (1, dp_size)

                target_shard_sizes = (1, gangs.sdp.size)
                target_shard_ranks = (0, gangs.sdp.rank)

        # Load the checkpoint files.
        pp_shards = []

        for tp_files in pp_files:
            tp_shards = []

            for dp_files in tp_files:
                dp_shards = []

                for dp_file in dp_files:
                    try:
                        dp_shard = self._tensor_loader.load(
                            dp_file, map_location=CPU, mmap=mmap, restrict=restrict
                        )
                    except TensorFileError as ex:
                        msg = f"{dp_file} is not a valid checkpoint file."

                        raise ModelCheckpointError(path, msg) from ex

                    if state_dict_converter is not None:
                        dp_shard = state_dict_converter(dp_shard)

                    dp_shards.append(dp_shard)

                tp_shards.append(dp_shards)

            pp_shards.append(tp_shards)

        # Reshard and yield the checkpoint tensors.
        for tp_shards in pp_shards:
            memo = set()

            # Assume that the very first data parallel shard contains all the
            # checkpoint keys.
            keys = list(tp_shards[0][0].keys())

            for key in keys:
                splits = []

                for dp_shards in tp_shards:
                    dp_splits = []

                    for dp_shard in dp_shards:
                        try:
                            dp_split = dp_shard.pop(key)
                        except KeyError:
                            break  # data parallel sharding can be uneven.

                        if not isinstance(dp_split, Tensor):
                            msg = f"{key} in {path} is not a `{Tensor}`."

                            raise ModelCheckpointError(path, msg)

                        dp_splits.append(dp_split)

                    splits.append(dp_splits)

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

    def _get_checkpoint_files(self, path: Path) -> list[list[list[Path]]]:
        pp_files = []

        for pp_idx in count():
            pp_dir = path.joinpath(f"pp_{pp_idx:02d}")

            is_dir = self._file_system.is_dir(pp_dir)
            if not is_dir:
                break

            tp_files = []

            for tp_idx in count():
                tp_dir = pp_dir.joinpath(f"tp_{tp_idx:02d}")

                is_dir = self._file_system.is_dir(tp_dir)
                if not is_dir:
                    break

                dp_files = []

                for dp_idx in count():
                    dp_file = tp_dir.joinpath(f"sdp_{dp_idx:02d}.pt")

                    is_file = self._file_system.is_file(dp_file)
                    if not is_file:
                        break

                    dp_files.append(dp_file)

                tp_files.append(dp_files)

            pp_files.append(tp_files)

        if not pp_files or not pp_files[0] or not pp_files[0][0]:
            msg = f"{path} does not contain any tensor files."

            raise ModelCheckpointError(path, msg)

        tp_size = len(pp_files[0])
        dp_size = len(pp_files[0][0])

        for pp_idx, tp_files in enumerate(pp_files):
            if len(tp_files) != tp_size:
                msg = f"Number of tensor parallel shards is expected to be {tp_size}, but the pipeline parallel shard at index {pp_idx} has {len(tp_files)} tensor parallel shards."

                raise ModelCheckpointError(path, msg)

            for tp_idx, dp_files in enumerate(tp_files):
                if len(dp_files) != dp_size:
                    msg = f"Number of data parallel shards is expected to be {dp_size}, but the tensor parallel shard at index {pp_idx}.{tp_idx} has {len(dp_files)} data parallel shards."

                    raise ModelCheckpointError(path, msg)

        return pp_files

    @override
    def supports_path(self, path: Path) -> bool:
        is_dir = self._file_system.is_dir(path)
        if not is_dir:
            return False

        pp_dir = path.joinpath("pp_00")

        return self._file_system.is_dir(pp_dir)
