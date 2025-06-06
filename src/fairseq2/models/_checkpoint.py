# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from itertools import count
from pathlib import Path
from typing import Protocol, final

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.device import CPU
from fairseq2.error import ContractError
from fairseq2.file_system import FileSystem
from fairseq2.gang import Gangs
from fairseq2.models.utils.sharder import ShardSpec
from fairseq2.utils.io import (
    HuggingFaceSafetensorsLoader,
    SafetensorsLoader,
    TensorLoader,
    TensorLoadError,
    TorchTensorLoader,
)


class CheckpointProcessor(Protocol):
    def __call__(self, checkpoint: dict[str, object]) -> dict[str, object]: ...


class CheckpointLoader(ABC):
    @abstractmethod
    def load(
        self,
        path: Path,
        gangs: Gangs,
        *,
        restrict: bool = True,
        processor: CheckpointProcessor | None = None,
        shard_specs: Mapping[str, ShardSpec] | None = None,
    ) -> Iterable[tuple[str, Tensor]]: ...

    @abstractmethod
    def supports_path(self, path: Path) -> bool: ...


@final
class DelegatingCheckpointLoader(CheckpointLoader):
    _loaders: Iterable[CheckpointLoader]

    def __init__(self, loaders: Iterable[CheckpointLoader]) -> None:
        self._loaders = loaders

    @override
    def load(
        self,
        path: Path,
        gangs: Gangs,
        *,
        restrict: bool = True,
        processor: CheckpointProcessor | None = None,
        shard_specs: Mapping[str, ShardSpec] | None = None,
    ) -> Iterable[tuple[str, Tensor]]:
        for loader in self._loaders:
            if loader.supports_path(path):
                return loader.load(
                    path,
                    gangs,
                    restrict=restrict,
                    processor=processor,
                    shard_specs=shard_specs,
                )

        raise CheckpointError(
            f"The '{path}' path does not point to any known checkpoint formats."
        )

    @override
    def supports_path(self, path: Path) -> bool:
        for loader in self._loaders:
            if loader.supports_path(path):
                return True

        return False


@final
class BasicCheckpointLoader(CheckpointLoader):
    _file_system: FileSystem
    _tensor_loader: TensorLoader

    def __init__(self, file_system: FileSystem, tensor_loader: TensorLoader) -> None:
        self._file_system = file_system

        self._tensor_loader = tensor_loader

    @override
    def load(
        self,
        path: Path,
        gangs: Gangs,
        *,
        restrict: bool = True,
        processor: CheckpointProcessor | None = None,
        shard_specs: Mapping[str, ShardSpec] | None = None,
    ) -> Iterable[tuple[str, Tensor]]:
        try:
            checkpoint = self._tensor_loader.load(
                path, map_location=CPU, restrict=restrict, mmap=True
            )
        except (FileNotFoundError, TensorLoadError) as ex:
            raise CheckpointError(
                f"The '{path}' tensor file cannot be loaded. See the nested exception for details."
            ) from ex

        if processor is not None:
            checkpoint = processor(checkpoint)

        source_shard_sizes = (1, 1)

        target_shard_sizes = (gangs.tp.size, gangs.sdp.size)
        target_shard_ranks = (gangs.tp.rank, gangs.sdp.rank)

        memo = set()

        for key, tensor in checkpoint.items():
            if not isinstance(tensor, Tensor):
                raise CheckpointError(
                    f"The value of the '{key}' key in the '{path}' checkpoint is not a `{Tensor}`."
                )

            if tensor in memo:  # Yield shared tensors only once.
                continue

            memo.add(tensor)

            tp_shards = [[tensor]]  # tp, dp

            tensor = _reshard_tensor(
                key,
                tp_shards,
                source_shard_sizes,
                target_shard_sizes,
                target_shard_ranks,
                shard_specs,
            )

            yield key, tensor

    @override
    def supports_path(self, path: Path) -> bool:
        if not path.suffix in (".pt", ".pth", ".bin"):
            return False

        try:
            return self._file_system.is_file(path)
        except OSError as ex:
            raise _access_error(path) from ex


@final
class SafetensorsCheckpointLoader(CheckpointLoader):
    _file_system: FileSystem
    _safetensors_loader: SafetensorsLoader

    def __init__(
        self, file_system: FileSystem, safetensors_loader: SafetensorsLoader
    ) -> None:
        self._file_system = file_system

        self._safetensors_loader = safetensors_loader

    @override
    def load(
        self,
        path: Path,
        gangs: Gangs,
        *,
        restrict: bool = True,
        processor: CheckpointProcessor | None = None,
        shard_specs: Mapping[str, ShardSpec] | None = None,
    ) -> Iterable[tuple[str, Tensor]]:
        try:
            is_dir = self._file_system.is_dir(path)
        except OSError as ex:
            raise _access_error(path) from ex

        if is_dir:
            try:
                files = list(self._file_system.glob(path, "*.safetensors"))
            except OSError as ex:
                raise _access_error(path) from ex

            if not files:
                raise CheckpointError(
                    f"No Safetensors file found under the '{path}' directory."
                )
        else:
            files = [path]

        source_shard_sizes = (1, 1)

        target_shard_sizes = (gangs.tp.size, gangs.sdp.size)
        target_shard_ranks = (gangs.tp.rank, gangs.sdp.rank)

        for file in files:
            try:
                checkpoint = self._safetensors_loader.load(file, device=CPU)
            except (FileNotFoundError, TensorLoadError) as ex:
                raise CheckpointError(
                    f"The '{file}' Safetensors file cannot be loaded. See the nested exception for details."
                ) from ex

            if processor is not None:
                checkpoint = processor(checkpoint)

            memo = set()

            for key, tensor in checkpoint.items():
                if not isinstance(tensor, Tensor):
                    raise ContractError(
                        f"The value of the '{key}' key in the '{path}' checkpoint is not a `{Tensor}`."
                    )

                if tensor in memo:  # Yield shared tensors only once.
                    continue

                memo.add(tensor)

                tp_shards = [[tensor]]  # tp, dp

                tensor = _reshard_tensor(
                    key,
                    tp_shards,
                    source_shard_sizes,
                    target_shard_sizes,
                    target_shard_ranks,
                    shard_specs,
                )

                yield key, tensor

    @override
    def supports_path(self, path: Path) -> bool:
        if path.suffix == ".safetensors":
            try:
                is_file = self._file_system.is_file(path)
            except OSError as ex:
                raise _access_error(path) from ex

            if is_file:
                return True

        try:
            is_dir = self._file_system.is_dir(path)
        except OSError as ex:
            raise _access_error(path) from ex

        if not is_dir:
            return False

        try:
            for file in self._file_system.glob(path, "*.safetensors"):
                if self._file_system.is_file(file):
                    return True
        except OSError as ex:
            raise _access_error(path) from ex

        return False


@final
class ShardedCheckpointLoader(CheckpointLoader):
    _file_system: FileSystem
    _tensor_loader: TensorLoader

    def __init__(self, file_system: FileSystem, tensor_loader: TensorLoader) -> None:
        self._file_system = file_system

        self._tensor_loader = tensor_loader

    @override
    def load(
        self,
        path: Path,
        gangs: Gangs,
        *,
        restrict: bool = True,
        processor: CheckpointProcessor | None = None,
        shard_specs: Mapping[str, ShardSpec] | None = None,
    ) -> Iterable[tuple[str, Tensor]]:
        try:
            is_dir = self._file_system.is_dir(path)
        except OSError as ex:
            raise _access_error(path) from ex

        if not is_dir:
            raise CheckpointError(
                f"The '{path}' path does not point to a fairseq2 model checkpoint."
            )

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
        pp_checkpoints = []

        for tp_files in pp_files:
            tp_checkpoints = []

            for dp_files in tp_files:
                dp_checkpoints = []

                for dp_file in dp_files:
                    try:
                        dp_checkpoint = self._tensor_loader.load(
                            dp_file, restrict=restrict, mmap=True
                        )
                    except (FileNotFoundError, TensorLoadError) as ex:
                        raise CheckpointError(
                            f"The '{dp_file}' tensor file cannot be loaded. See the nested exception for details."
                        ) from ex

                    if processor is not None:
                        dp_checkpoint = processor(dp_checkpoint)

                    dp_checkpoints.append(dp_checkpoint)

                tp_checkpoints.append(dp_checkpoints)

            pp_checkpoints.append(tp_checkpoints)

        # Reshard and yield the checkpoint tensors.
        for tp_checkpoints in pp_checkpoints:
            memo = set()

            # Assume that the very first data parallel shard contains all the
            # checkpoint keys.
            keys = list(tp_checkpoints[0][0].keys())

            for key in keys:
                tp_shards = []

                for dp_checkpoints in tp_checkpoints:
                    dp_shards = []

                    for dp_checkpoint in dp_checkpoints:
                        try:
                            dp_shard = dp_checkpoint.pop(key)
                        except KeyError:
                            break  # data parallel sharding can be uneven.

                        if not isinstance(dp_shard, Tensor):
                            raise CheckpointError(
                                f"The value of the '{key}' key in the '{path}' checkpoint is not a `{Tensor}`."
                            )

                        dp_shards.append(dp_shard)

                    tp_shards.append(dp_shards)

                dp_shard_0 = tp_shards[0][0]

                if dp_shard_0 in memo:  # Yield shared tensors only once.
                    continue

                memo.add(dp_shard_0)

                tensor = _reshard_tensor(
                    key,
                    tp_shards,
                    source_shard_sizes,
                    target_shard_sizes,
                    target_shard_ranks,
                    shard_specs,
                )

                yield key, tensor

    def _get_checkpoint_files(self, path: Path) -> list[list[list[Path]]]:
        pp_files = []

        for pp_idx in count():
            pp_dir = path.joinpath(f"pp_{pp_idx:02d}")

            try:
                is_dir = self._file_system.is_dir(pp_dir)
            except OSError as ex:
                raise _access_error(path) from ex

            if not is_dir:
                break

            tp_files = []

            for tp_idx in count():
                tp_dir = path.joinpath(f"tp_{tp_idx:02d}")

                try:
                    is_dir = self._file_system.is_dir(tp_dir)
                except OSError as ex:
                    raise _access_error(path) from ex

                if not is_dir:
                    break

                dp_files = []

                for dp_idx in count():
                    dp_file = path.joinpath(f"sdp_{dp_idx:02d}")

                    try:
                        is_file = self._file_system.is_file(dp_file)
                    except OSError as ex:
                        raise _access_error(path) from ex

                    if not is_file:
                        break

                    dp_files.append(dp_file)

                tp_files.append(dp_files)

            pp_files.append(tp_files)

        if not pp_files or not pp_files[0] or not pp_files[0][0]:
            raise CheckpointError(
                f"The '{path}' directory does not contain any tensor files.",
            )

        tp_size = len(pp_files[0])
        dp_size = len(pp_files[0][0])

        for tp_files in pp_files:
            if len(tp_files) != tp_size:
                raise CheckpointError("invalid")

            for dp_files in tp_files:
                if len(dp_files) != dp_size:
                    raise CheckpointError("invalid")

        return pp_files

    @override
    def supports_path(self, path: Path) -> bool:
        try:
            is_dir = self._file_system.is_dir(path)
        except OSError as ex:
            raise _access_error(path) from ex

        if not is_dir:
            return False

        pp_dir = path.joinpath("pp_00")

        try:
            return self._file_system.is_dir(pp_dir)
        except OSError as ex:
            raise _access_error(path) from ex


def _reshard_tensor(
    key: str,
    source_tp_shards: list[list[Tensor]],
    source_shard_sizes: tuple[int, int],
    target_shard_sizes: tuple[int, int],
    target_shard_ranks: tuple[int, int],
    shard_specs: Mapping[str, ShardSpec] | None,
) -> Tensor:
    source_tp_size, source_dp_size = source_shard_sizes
    target_tp_size, target_dp_size = target_shard_sizes

    target_tp_rank, target_dp_rank = target_shard_ranks

    # If the source and target tensor parallel sizes match, we can directly
    # return the unsharded data parallel tensor.
    if source_tp_size == target_tp_size:
        source_dp_shards = source_tp_shards[target_tp_rank]

        if source_dp_size == 1:
            return source_dp_shards[0]

        return torch.cat(source_dp_shards, dim=0)

    tp_dim = _get_tp_dim(key, shard_specs)

    # We assume that non-tensor parallel parameters are always replicated.
    if tp_dim == -1:
        source_dp_shards = source_tp_shards[0]

        if source_dp_size == 1:
            return source_dp_shards[0]

        return torch.cat(source_dp_shards, dim=0)

    tp_shards = []

    # Unshard the tensor over the tensor parallel dimension.
    for source_dp_shards in source_tp_shards:
        if source_dp_size == 1:
            tp_shard = source_dp_shards[0]
        else:
            tp_shard = torch.cat(source_dp_shards, dim=0)

        tp_shards.append(tp_shard)

    if source_tp_size == 1:
        tensor = tp_shards[0]
    else:
        tensor = torch.cat(tp_shards, dim=tp_dim)

    del tp_shards

    if target_tp_size == 1:
        return tensor

    # Reshard with the target tensor parallel size.
    target_tp_shards = tensor.chunk(target_tp_size, dim=tp_dim)

    return target_tp_shards[target_tp_rank]


def _get_tp_dim(key: str, shard_specs: Mapping[str, ShardSpec] | None) -> int:
    if shard_specs is None:
        return -1

    offset = key.rfind(".")
    if offset >= 0:
        module_name = key[:offset]
    else:
        module_name = key

    for pattern, spec in shard_specs.items():
        if re.match(pattern, module_name):
            return spec.dim

    return -1


class CheckpointError(Exception):
    pass


def _access_error(path: Path) -> CheckpointError:
    return CheckpointError(
        f"The '{path}' path cannot be accessed. See the nested exception for details."
    )


def create_model_checkpoint_loader(file_system: FileSystem) -> CheckpointLoader:
    tensor_loader = TorchTensorLoader(file_system)

    safetensors_loader = HuggingFaceSafetensorsLoader(file_system)

    basic_loader = BasicCheckpointLoader(file_system, tensor_loader)

    st_loader = SafetensorsCheckpointLoader(file_system, safetensors_loader)

    native_loader = ShardedCheckpointLoader(file_system, tensor_loader)

    return DelegatingCheckpointLoader([basic_loader, st_loader, native_loader])
