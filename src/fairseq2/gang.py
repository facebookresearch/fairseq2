# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import final

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup, ReduceOp  # type: ignore[attr-defined]

from fairseq2.typing import Device, finaloverride


class ReduceOperation(Enum):
    """Specifies a reduce operation."""

    SUM = 1
    MEAN = 2
    PRODUCT = 3
    MIN = 4
    MAX = 5


class Gang(ABC):
    """Represents a set of processes that work collectively."""

    rank: int
    size: int
    device: Device

    def __init__(self, rank: int, size: int, device: Device) -> None:
        """
        :param rank:
            The rank of this process in the gang.
        :param size:
            The number of processes that are part of the gang.
        :param device:
            The associated device.
        """
        self.rank = rank
        self.size = size

        self.device = device

    @abstractmethod
    def as_process_group(self) -> ProcessGroup:
        """Return this gang as a process group."""

    @abstractmethod
    def barrier(self) -> None:
        """Synchronize all ranks."""

    @abstractmethod
    def all_reduce(self, tensor: Tensor, op: ReduceOperation) -> None:
        """Reduce the tensor across all ranks.

        :param tensor:
            The input and output tensor of the operation.
        :param op:
            The element-wise reduce operation.
        """

    @abstractmethod
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor) -> None:
        """Gather tensors from all ranks and put them in a single tensor.

        :param output_tensor:
            The output tensor to accomodate tensor elements from all ranks.
        :param input_tensor:
            The tensor to be gathered from the current rank.
        """


@final
class FakeGang(Gang):
    """Represents a non-distributed gang for local use."""

    def __init__(self, device: Device) -> None:
        super().__init__(rank=0, size=1, device=device)

    @finaloverride
    def as_process_group(self) -> ProcessGroup:
        raise RuntimeError("`FakeGang` does not support conversion to process group.")

    @finaloverride
    def barrier(self) -> None:
        pass

    @finaloverride
    def all_reduce(self, tensor: Tensor, op: ReduceOperation) -> None:
        pass

    @finaloverride
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor) -> None:
        output_tensor.copy_(input_tensor)


@final
class ProcessGroupGang(Gang):
    """Represents a gang that wraps a process group."""

    pg: ProcessGroup

    @staticmethod
    def from_process_group(pg: ProcessGroup, device: Device) -> Gang:
        """Wrap ``pg`` as a gang.

        :param pg:
            The process group to wrap.
        :param device:
            The associated device.
        """
        return ProcessGroupGang(pg, device)

    @staticmethod
    def from_default_process_group() -> Gang:
        """Wrap the default process group as a gang.

        The default process group will be initialized as part of this function
        if it is not already initialized. If CUDA is available, NCCL; otherwise,
        Gloo backend will be used.
        """
        if not dist.is_available():
            raise RuntimeError("`torch.distributed` is not available.")

        if dist.is_initialized():
            backend = dist.get_backend()

            if backend == "gloo":
                device = Device("cpu")
            elif backend == "nccl":
                device = _determine_default_cuda_device()
            else:
                raise RuntimeError(
                    f"Only `nccl` and `gloo` backends are supported for device selection, but the process group uses the `{backend}` backend."
                )
        else:
            device = _determine_default_device()

            dist.init_process_group("nccl" if device.type == "cuda" else "gloo")

        if dist.group.WORLD is None:
            raise RuntimeError(
                "The default process group is not available. Please file a bug report."
            )

        return ProcessGroupGang(dist.group.WORLD, device)

    def __init__(self, pg: ProcessGroup, device: Device) -> None:
        super().__init__(dist.get_rank(pg), dist.get_world_size(pg), device)

        self.pg = pg

    @finaloverride
    def as_process_group(self) -> ProcessGroup:
        return self.pg

    @finaloverride
    def barrier(self) -> None:
        dist.barrier(group=self.pg)

    @finaloverride
    def all_reduce(self, tensor: Tensor, op: ReduceOperation) -> None:
        dist.all_reduce(tensor, self._get_reduce_op(op), group=self.pg)

    @finaloverride
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor) -> None:
        dist.all_gather_into_tensor(output_tensor, input_tensor, group=self.pg)

    @staticmethod
    def _get_reduce_op(op: ReduceOperation):  # type: ignore[no-untyped-def]
        if op == ReduceOperation.SUM:
            return ReduceOp.SUM
        if op == ReduceOperation.MEAN:
            return ReduceOp.AVG  # type: ignore[attr-defined]
        if op == ReduceOperation.PRODUCT:
            return ReduceOp.PRODUCT
        if op == ReduceOperation.MIN:
            return ReduceOp.MIN
        if op == ReduceOperation.MAX:
            return ReduceOp.MAX

        raise ValueError(
            f"`op` must be an operation supported by the underlying process group, but is `{op}` instead."
        )


def _determine_default_device() -> Device:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return _determine_default_cuda_device()

    return Device("cpu")


def _determine_default_cuda_device() -> Device:
    num_devices = torch.cuda.device_count()

    # We use the `LOCAL_RANK` environment variable to determine which GPU to
    # pick in case the process has more than one GPU available.
    local_rank_env = os.getenv("LOCAL_RANK")
    if local_rank_env is None:
        if num_devices > 1:
            raise RuntimeError(
                f"The default device cannot be determined. There are {num_devices} GPUs available, but the `LOCAL_RANK` environment variable is not set."
            )

        return Device("cuda", index=0)

    try:
        local_rank = int(local_rank_env)
    except ValueError:
        raise RuntimeError(
            f"The value of the `LOCAL_RANK` environment variable must be an integer, but is '{local_rank_env}' instead."
        )

    if local_rank >= num_devices:
        raise RuntimeError(
            f"The value of the `LOCAL_RANK` environment variable must be less than the number of available GPUs ({num_devices}), but is {local_rank} instead."
        )

    device = Device("cuda", index=local_rank)

    # As of PyTorch 2.0, FSDP fails to work if the default device is not set.
    torch.cuda.set_device(device)

    return device
