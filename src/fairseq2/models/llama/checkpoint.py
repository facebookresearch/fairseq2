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
from fairseq2.model_checkpoint import (
    BadModelCheckpointError,
    ModelCheckpointLoader,
    ModelCheckpointLoadOptions,
    reshard_tensor,
)


@final
class _LLaMACheckpointLoader(ModelCheckpointLoader):
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

        tp_files = self._get_checkpoint_files(path)

        tp_size = len(tp_files)

        gangs = options.gangs or self._gang_context.get_current_gangs()

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

        load_options = TensorFileLoadOptions(
            map_location=CPU, mmap=options.mmap, restrict=options.restrict
        )

        # Load the checkpoint files.
        tp_shards = []

        for tp_file in tp_files:
            try:
                tp_shard = self._tensor_file_loader.load(tp_file, load_options)
            except (PickleError, EOFError) as ex:
                raise BadModelCheckpointError(
                    f"'{tp_file}' is not a valid PyTorch tensor file"
                ) from ex
            except OSError as ex:
                raise OSError(
                    f"an I/O error occurred while reading checkpoint file '{tp_file}'"
                ) from ex

            if options.state_dict_converter is not None:
                tp_shard = options.state_dict_converter(tp_shard)

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
                    from fairseq2.typing import get_full_type_name as n

                    raise BadModelCheckpointError(
                        f"key '{key}' in checkpoint '{path}' is expected to be of type `Tensor`, but is of type `{n(tp_split)}` instead"  # fmt: skip
                    )

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
                shard_dims,
            )

            yield key, tensor

            del tensor

    def _get_checkpoint_files(self, path: Path) -> list[Path]:
        tp_files = []

        for tp_idx in count():
            tp_file = path.joinpath(f"consolidated.{tp_idx:02d}.pth")

            try:
                is_file = self._file_system.is_file(tp_file)
            except OSError as ex:
                self._raise_path_access_error(tp_file, ex)

            if not is_file:
                break

            tp_files.append(tp_file)

        if not tp_files:
            raise BadModelCheckpointError(
                f"checkpoint '{path}' does not have a known format"
            )

        return tp_files

    @override
    def supports_path(self, path: Path) -> bool:
        try:
            is_dir = self._file_system.is_dir(path)
        except OSError as ex:
            self._raise_path_access_error(path, ex)

        if not is_dir:
            return False

        file = path.joinpath("consolidated.00.pth")

        try:
            return self._file_system.is_file(file)
        except OSError as ex:
            self._raise_path_access_error(file, ex)

    def _raise_path_access_error(self, path: Path, cause: OSError) -> NoReturn:
        raise OSError(f"an I/O error occurred while accessing path '{path}'") from cause
