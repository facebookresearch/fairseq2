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
        mmap: bool = False,
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
        mmap: bool = False,
        restrict: bool = True,
        processor: CheckpointProcessor | None = None,
        shard_specs: Mapping[str, ShardSpec] | None = None,
    ) -> Iterable[tuple[str, Tensor]]:
        for loader in self._loaders:
            if loader.supports_path(path):
                return loader.load(
                    path,
                    gangs,
                    mmap=mmap,
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
        mmap: bool = False,
        restrict: bool = True,
        processor: CheckpointProcessor | None = None,
        shard_specs: Mapping[str, ShardSpec] | None = None,
    ) -> Iterable[tuple[str, Tensor]]:
        try:
            checkpoint = self._tensor_loader.load(
                path, map_location=CPU, mmap=mmap, restrict=restrict
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
                    f"The value of the '{key}' key in the '{path}' file is not a `{Tensor}`."
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

            del tensor

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
        mmap: bool = False,
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

        checkpoint = {}

        for file in files:
            try:
                st_shard = self._safetensors_loader.load(file, device=CPU, mmap=mmap)
            except (FileNotFoundError, TensorLoadError) as ex:
                raise CheckpointError(
                    f"The '{file}' Safetensors file cannot be loaded. See the nested exception for details."
                ) from ex

            for key, value in st_shard.items():
                if key in checkpoint:
                    raise CheckpointError(
                        f"The '{path}' directory has more than one Safetensors file with the key '{key}'."
                    )

                checkpoint[key] = value

        if processor is not None:
            checkpoint = processor(checkpoint)

        source_shard_sizes = (1, 1)

        target_shard_sizes = (gangs.tp.size, gangs.sdp.size)
        target_shard_ranks = (gangs.tp.rank, gangs.sdp.rank)

        memo = set()

        for key, tensor in checkpoint.items():
            if not isinstance(tensor, Tensor):
                raise ContractError(
                    f"The value of the '{key}' key in the '{path}' file is not a `{Tensor}`."
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

            del tensor

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
        mmap: bool = False,
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
                f"The '{path}' path does not point to a fairseq2 checkpoint."
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
                            dp_file, map_location=CPU, mmap=mmap, restrict=restrict
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
                                f"The value of the '{key}' key in the '{path}' file is not a `{Tensor}`."
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

                del tensor

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
                tp_dir = pp_dir.joinpath(f"tp_{tp_idx:02d}")

                try:
                    is_dir = self._file_system.is_dir(tp_dir)
                except OSError as ex:
                    raise _access_error(path) from ex

                if not is_dir:
                    break

                dp_files = []

                for dp_idx in count():
                    dp_file = tp_dir.joinpath(f"sdp_{dp_idx:02d}.pt")

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

        for pp_idx, tp_files in enumerate(pp_files):
            if len(tp_files) != tp_size:
                raise CheckpointError(
                    f"The number of tensor parallel shards is expected to be {tp_size}, but the pipeline parallel shard at index {pp_idx} has {len(tp_files)} tensor parallel shards."
                )

            for tp_idx, dp_files in enumerate(tp_files):
                if len(dp_files) != dp_size:
                    raise CheckpointError(
                        f"The number of data parallel shards is expected to be {dp_size}, but the tensor parallel shard at index {pp_idx}.{tp_idx} has {len(dp_files)} data parallel shards."
                    )

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


@final
class LLaMACheckpointLoader(CheckpointLoader):
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
        mmap: bool = False,
        restrict: bool = True,
        processor: CheckpointProcessor | None = None,
        shard_specs: Mapping[str, ShardSpec] | None = None,
    ) -> Iterable[tuple[str, Tensor]]:
        # Handle legacy paths with format specifiers.
        if "shard_idx" in path.name:
            path = path.parent

        try:
            is_dir = self._file_system.is_dir(path)
        except OSError as ex:
            raise _access_error(path) from ex

        if not is_dir:
            raise CheckpointError(
                f"The '{path}' path does not point to a LLaMA checkpoint."
            )

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
        tp_checkpoints = []

        for tp_file in tp_files:
            try:
                tp_checkpoint = self._tensor_loader.load(
                    tp_file, map_location=CPU, mmap=mmap, restrict=restrict
                )
            except (FileNotFoundError, TensorLoadError) as ex:
                raise CheckpointError(
                    f"The '{tp_file}' tensor file cannot be loaded. See the nested exception for details."
                ) from ex

            if processor is not None:
                tp_checkpoint = processor(tp_checkpoint)

            tp_checkpoints.append(tp_checkpoint)

        memo = set()

        # Assume that the very first tensor parallel shard contains all the
        # checkpoint keys.
        keys = list(tp_checkpoints[0].keys())

        for key in keys:
            tp_shards = []

            for tp_checkpoint in tp_checkpoints:
                try:
                    tp_shard = tp_checkpoint.pop(key)
                except KeyError:
                    break  # data parallel sharding can be uneven.

                if not isinstance(tp_shard, Tensor):
                    raise CheckpointError(
                        f"The value of the '{key}' key in the '{path}' file is not a `{Tensor}`."
                    )

                tp_shards.append([tp_shard])

            tp_shard_0 = tp_shards[0][0]

            if tp_shard_0 in memo:  # Yield shared tensors only once.
                continue

            memo.add(tp_shard_0)

            tensor = _reshard_tensor(
                key,
                tp_shards,
                source_shard_sizes,
                target_shard_sizes,
                target_shard_ranks,
                shard_specs,
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
                raise _access_error(path) from ex

            if not is_file:
                break

            tp_files.append(tp_file)

        if not tp_files:
            raise CheckpointError(
                f"The '{path}' directory does not contain any tensor files.",
            )

        return tp_files

    @override
    def supports_path(self, path: Path) -> bool:
        # Handle legacy paths with format specifiers.
        if "shard_idx" in path.name:
            path = path.parent

        try:
            is_dir = self._file_system.is_dir(path)
        except OSError as ex:
            raise _access_error(path) from ex

        if not is_dir:
            return False

        file = path.joinpath("consolidated.00.pth")

        try:
            return self._file_system.is_file(file)
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

    # Unshard the tensor over the source tensor parallel dimension.
    for source_dp_shards in source_tp_shards:
        if source_dp_size == 1:
            tp_shard = source_dp_shards[0]
        else:
            tp_shard = torch.cat(source_dp_shards, dim=0)

        tp_shards.append(tp_shard)

    # Reshard the tensor over the target parallel dimension.
    source_tp_dim_size = tp_shards[0].size(tp_dim)

    tp_dim_size = source_tp_dim_size * source_tp_size

    target_tp_dim_size = tp_dim_size // target_tp_size

    f_target_idx = target_tp_rank * target_tp_dim_size
    l_target_idx = target_tp_rank * target_tp_dim_size + target_tp_dim_size - 1

    f_source_tp_shard_idx = f_target_idx // source_tp_dim_size
    l_source_tp_shard_idx = l_target_idx // source_tp_dim_size

    f_source_idx = f_source_tp_shard_idx * source_tp_dim_size

    tp_sub_shards = []

    for idx in range(f_source_tp_shard_idx, l_source_tp_shard_idx + 1):
        tp_sub_shards.append(tp_shards[idx])

    del tp_shards

    tensor = torch.cat(tp_sub_shards, dim=tp_dim)

    del tp_sub_shards

    return tensor.narrow(
        dim=tp_dim, start=f_target_idx - f_source_idx, length=target_tp_dim_size
    )


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


def create_checkpoint_loader(file_system: FileSystem) -> CheckpointLoader:
    tensor_loader = TorchTensorLoader(file_system)

    safetensors_loader = HuggingFaceSafetensorsLoader(file_system)

    basic_loader = BasicCheckpointLoader(file_system, tensor_loader)

    st_loader = SafetensorsCheckpointLoader(file_system, safetensors_loader)

    llama_loader = LLaMACheckpointLoader(file_system, tensor_loader)

    native_loader = ShardedCheckpointLoader(file_system, tensor_loader)

    return DelegatingCheckpointLoader(
        [basic_loader, st_loader, llama_loader, native_loader]
    )
