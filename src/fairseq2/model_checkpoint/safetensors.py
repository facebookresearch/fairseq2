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
from fairseq2.error import InternalError
from fairseq2.file_system import FileSystem
from fairseq2.gang import Gangs
from fairseq2.io import SafetensorsLoader, TensorFileError
from fairseq2.model_checkpoint.common import reshard_tensor
from fairseq2.model_checkpoint.loader import (
    ModelCheckpointError,
    ModelCheckpointLoader,
    StateDictConverter,
)
from fairseq2.sharder import ShardSpec


@final
class SafetensorsCheckpointLoader(ModelCheckpointLoader):
    """
    Loads Safetensors checkpoints.

    This loader supports both single-file and multi-file Safetensors checkpoints
    where multi-file checkpoints typically follow the "model-x-of-N.safetensors"
    pattern as in Hugging Face Hub.
    """

    def __init__(
        self, file_system: FileSystem, safetensors_loader: SafetensorsLoader
    ) -> None:
        self._file_system = file_system
        self._safetensors_loader = safetensors_loader

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
        is_dir = self._file_system.is_dir(path)
        if is_dir:
            files = list(self._file_system.glob(path, "*.safetensors"))
            if not files:
                msg = f"No checkpoint files found under {path}."

                raise ModelCheckpointError(path, msg)
        else:
            files = [path]

        checkpoint = {}

        for file in files:
            try:
                st_shard = self._safetensors_loader.load(file, device=CPU, mmap=mmap)
            except TensorFileError as ex:
                msg = f"{file} is not a valid checkpoint file."

                raise ModelCheckpointError(path, msg) from ex

            for key, value in st_shard.items():
                if key in checkpoint:
                    msg = f"{path} has more than one checkpoint file with key {key}."

                    raise ModelCheckpointError(path, msg)

                checkpoint[key] = value

        if state_dict_converter is not None:
            checkpoint = state_dict_converter(checkpoint)

        source_shard_sizes = (1, 1)

        target_shard_sizes = (gangs.tp.size, gangs.sdp.size)
        target_shard_ranks = (gangs.tp.rank, gangs.sdp.rank)

        memo = set()

        for key, tensor in checkpoint.items():
            if not isinstance(tensor, Tensor):
                raise InternalError(f"{key} in {path} is not a `{Tensor}`.")

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
        if path.suffix == ".safetensors":
            is_file = self._file_system.is_file(path)
            if is_file:
                return True

        is_dir = self._file_system.is_dir(path)
        if not is_dir:
            return False

        for file in self._file_system.glob(path, "*.safetensors"):
            if self._file_system.is_file(file):
                return True

        return False
