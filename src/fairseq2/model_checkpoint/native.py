# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import count
from pathlib import Path
from pickle import PickleError
from typing import NoReturn, final

from torch import Tensor
from typing_extensions import override

from fairseq2.device import CPU
from fairseq2.file_system import FileSystem, _raise_file_not_found_error
from fairseq2.gang import GangContext
from fairseq2.io import TensorFileLoader, TensorFileLoadOptions
from fairseq2.model_checkpoint.common import reshard_tensor
from fairseq2.model_checkpoint.loader import (
    BadModelCheckpointError,
    ModelCheckpointLoader,
    ModelCheckpointLoadOptions,
)


@final
class _NativeModelCheckpointLoader(ModelCheckpointLoader):
    """
    Loads native fairseq2 checkpoints.

    The native fairseq2 format is optimized for efficient storage and loading of
    model checkpoints in distributed configurations.
    """

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

        try:
            is_dir = self._file_system.is_dir(path)
        except OSError as ex:
            self._raise_path_access_error(path, ex)

        if not is_dir:
            try:
                path_exists = self._file_system.exists(path)
            except OSError as ex:
                self._raise_path_access_error(path, ex)

            if not path_exists:
                _raise_file_not_found_error(path)

            raise BadModelCheckpointError(
                f"checkpoint '{path}' does not have a known format"
            )

        pp_files = self._get_checkpoint_files(path)

        pp_size = len(pp_files)
        tp_size = len(pp_files[0])
        dp_size = len(pp_files[0][0])

        gangs = options.gangs or self._gang_context.get_current_gangs()

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

        load_options = TensorFileLoadOptions(
            map_location=CPU, mmap=options.mmap, restrict=options.restrict
        )

        # Load the checkpoint files.
        pp_shards = []

        for tp_files in pp_files:
            tp_shards = []

            for dp_files in tp_files:
                dp_shards = []

                for dp_file in dp_files:
                    try:
                        dp_shard = self._tensor_file_loader.load(dp_file, load_options)
                    except (PickleError, EOFError) as ex:
                        raise BadModelCheckpointError(
                            f"'{dp_file}' is not a valid PyTorch tensor file"
                        ) from ex
                    except OSError as ex:
                        raise OSError(
                            f"an I/O error occurred while reading checkpoint file '{dp_file}'"
                        ) from ex

                    if options.state_dict_converter is not None:
                        dp_shard = options.state_dict_converter(dp_shard)

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
                            from fairseq2.typing import get_full_type_name as n

                            raise BadModelCheckpointError(
                                f"key '{key}' in checkpoint '{path}' is expected to be of type `Tensor`, but is of type `{n(dp_split)}` instead"  # fmt: skip
                            )

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
                    shard_dims,
                )

                yield key, tensor

                del tensor

    def _get_checkpoint_files(self, path: Path) -> list[list[list[Path]]]:
        pp_files = []

        for pp_idx in count():
            pp_dir = path.joinpath(f"pp_{pp_idx:02d}")

            try:
                is_dir = self._file_system.is_dir(pp_dir)
            except OSError as ex:
                self._raise_path_access_error(pp_dir, ex)

            if not is_dir:
                break

            tp_files = []

            for tp_idx in count():
                tp_dir = pp_dir.joinpath(f"tp_{tp_idx:02d}")

                try:
                    is_dir = self._file_system.is_dir(tp_dir)
                except OSError as ex:
                    self._raise_path_access_error(tp_dir, ex)

                if not is_dir:
                    break

                dp_files = []

                for dp_idx in count():
                    dp_file = tp_dir.joinpath(f"sdp_{dp_idx:02d}.pt")

                    try:
                        is_file = self._file_system.is_file(dp_file)
                    except OSError as ex:
                        self._raise_path_access_error(dp_file, ex)

                    if not is_file:
                        break

                    dp_files.append(dp_file)

                tp_files.append(dp_files)

            pp_files.append(tp_files)

        if not pp_files or not pp_files[0] or not pp_files[0][0]:
            raise BadModelCheckpointError(
                f"checkpoint '{path}' does not have a known format"
            )

        tp_size = len(pp_files[0])
        dp_size = len(pp_files[0][0])

        for pp_idx, tp_files in enumerate(pp_files):
            if len(tp_files) != tp_size:
                raise BadModelCheckpointError(
                    f"each pipeline parallel shard in checkpoint '{path}' is expected to have {tp_size} tensor parallel shards, but the pipeline parallel shard at index {pp_idx} has {len(tp_files)} tensor parallel shards"  # fmt: skip
                )

            for tp_idx, dp_files in enumerate(tp_files):
                if len(dp_files) != dp_size:
                    raise BadModelCheckpointError(
                        f"each tensor parallel shard in checkpoint '{path}' is expected to have {dp_size} data parallel shards, but the tensor parallel shard at index {pp_idx}.{tp_idx} has {len(dp_files)} data parallel shards"  # fmt: skip
                    )

        return pp_files

    @override
    def supports_path(self, path: Path) -> bool:
        try:
            is_dir = self._file_system.is_dir(path)
        except OSError as ex:
            self._raise_path_access_error(path, ex)

        if not is_dir:
            return False

        pp_dir = path.joinpath("pp_00")

        try:
            return self._file_system.is_dir(pp_dir)
        except OSError as ex:
            self._raise_path_access_error(pp_dir, ex)

    def _raise_path_access_error(self, path: Path, cause: OSError) -> NoReturn:
        raise OSError(f"an I/O error occurred while accessing path '{path}'") from cause
