# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import NoReturn, final

from safetensors import SafetensorError
from torch import Tensor
from typing_extensions import override

from fairseq2.device import CPU
from fairseq2.error import InternalError
from fairseq2.file_system import FileSystem, _raise_file_not_found_error
from fairseq2.gang import GangContext
from fairseq2.io import SafetensorsLoader, SafetensorsLoadOptions
from fairseq2.model_checkpoint.common import reshard_tensor
from fairseq2.model_checkpoint.loader import (
    BadModelCheckpointError,
    ModelCheckpointLoader,
    ModelCheckpointLoadOptions,
)


@final
class _SafetensorsCheckpointLoader(ModelCheckpointLoader):
    """
    Loads Safetensors checkpoints.

    This loader supports both single-file and multi-file Safetensors checkpoints
    where multi-file checkpoints typically follow the "model-x-of-N.safetensors"
    pattern as in Hugging Face Hub.
    """

    def __init__(
        self,
        file_system: FileSystem,
        safetensors_loader: SafetensorsLoader,
        gang_context: GangContext,
    ) -> None:
        self._file_system = file_system
        self._safetensors_loader = safetensors_loader
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
            path_exists = self._file_system.exists(path)
        except OSError as ex:
            self._raise_path_access_error(path, ex)

        if not path_exists:
            _raise_file_not_found_error(path)

        try:
            is_dir = self._file_system.is_dir(path)
        except OSError as ex:
            self._raise_path_access_error(path, ex)

        if is_dir:
            try:
                files = list(self._file_system.glob(path, "*.safetensors"))
            except OSError as ex:
                raise OSError(
                    f"an I/O error occurred while globing directory '{path}'"
                ) from ex

            if not files:
                raise BadModelCheckpointError(
                    f"checkpoint '{path}' does not have a known format"
                )
        else:
            files = [path]

        checkpoint = {}

        load_options = SafetensorsLoadOptions(device=CPU, mmap=options.mmap)

        for file in files:
            try:
                shard = self._safetensors_loader.load(file, load_options)
            except (SafetensorError, EOFError) as ex:
                raise BadModelCheckpointError(
                    f"'{path}' is not a valid Safetensors file"
                ) from ex
            except OSError as ex:
                raise OSError(
                    f"an I/O error occurred while reading checkpoint file '{file}'"
                ) from ex

            for key, value in shard.items():
                if key in checkpoint:
                    raise BadModelCheckpointError(
                        f"more than one Safetensors file in checkpoint '{path}' contains key '{key}'"  # fmt: skip
                    )

                checkpoint[key] = value

        if options.state_dict_converter is not None:
            checkpoint = options.state_dict_converter(checkpoint)

        gangs = options.gangs or self._gang_context.get_current_gangs()

        source_shard_sizes = (1, 1)

        target_shard_sizes = (gangs.tp.size, gangs.sdp.size)
        target_shard_ranks = (gangs.tp.rank, gangs.sdp.rank)

        memo = set()

        for key, tensor in checkpoint.items():
            if not isinstance(tensor, Tensor):
                raise InternalError(
                    f"key '{key}' in checkpoint '{path}' is not a `Tensor`"
                )

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
        if path.suffix == ".safetensors":
            try:
                is_file = self._file_system.is_file(path)
            except OSError as ex:
                self._raise_path_access_error(path, ex)

            if is_file:
                return True

        try:
            is_dir = self._file_system.is_dir(path)
        except OSError as ex:
            self._raise_path_access_error(path, ex)

        if not is_dir:
            return False

        def files() -> Iterator[Path]:
            try:
                yield from self._file_system.glob(path, "*.safetensors")
            except OSError as ex:
                raise OSError(
                    f"an I/O error occurred while globing directory '{path}'"
                ) from ex

        for file in files():
            try:
                is_file = self._file_system.is_file(file)
            except OSError as ex:
                self._raise_path_access_error(file, ex)

            if is_file:
                return True

        return False

    def _raise_path_access_error(self, path: Path, cause: OSError) -> NoReturn:
        raise OSError(f"an I/O error occurred while accessing path '{path}'") from cause
