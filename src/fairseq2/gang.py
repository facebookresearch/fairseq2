# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from abc import ABC, abstractmethod
from datetime import timedelta
from enum import Enum
from typing import Optional, final

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup, ReduceOp

from fairseq2.typing import CPU, Device, override
from fairseq2.utils.logging import get_log_writer
from fairseq2.utils.version import _is_pt22_or_greater

log = get_log_writer(__name__)


class ReduceOperation(Enum):
    """Specifies a reduce operation."""

    SUM = 1
    MEAN = 2
    PRODUCT = 3
    MIN = 4
    MAX = 5


class Gang(ABC):
    """Represents a set of processes that work collectively."""

    @abstractmethod
    def close(self) -> None:
        """Closes and destroys the gang."""

    @abstractmethod
    def as_process_group(self) -> ProcessGroup:
        """Return this gang as a process group."""

    @abstractmethod
    def barrier(self) -> None:
        """Synchronize all processes."""

    @abstractmethod
    def all_reduce(self, tensor: Tensor, op: ReduceOperation) -> None:
        """Reduce the tensor across all processes.

        :param tensor:
            The input and output tensor of the operation.
        :param op:
            The element-wise reduce operation.
        """

    @abstractmethod
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor) -> None:
        """Gather tensors from all processes and put them in a single tensor.

        :param output_tensor:
            The output tensor to accomodate tensors from all processes.
        :param input_tensor:
            The tensor to be gathered from this process.
        """

    @property
    @abstractmethod
    def rank(self) -> int:
        """The rank of this process in the gang."""

    @property
    @abstractmethod
    def size(self) -> int:
        """The number of processes that are part of the gang."""

    @property
    @abstractmethod
    def device(self) -> Device:
        """The associated device."""


class AbstractGang(Gang):
    """Provides a skeletal implementation of :class:`Gang`."""

    _rank: int
    _size: int
    _device: Device

    def __init__(self, rank: int, size: int, device: Device) -> None:
        """
        :param rank:
            The rank of this process in the gang.
        :param size:
            The number of processes that are part of the gang.
        :param device:
            The associated device.
        """
        self._rank = rank
        self._size = size

        self._device = device

    @final
    @property
    @override
    def rank(self) -> int:
        return self._rank

    @final
    @property
    @override
    def size(self) -> int:
        return self._size

    @final
    @property
    @override
    def device(self) -> Device:
        return self._device


@final
class FakeGang(AbstractGang):
    """Represents a non-distributed gang for local use."""

    def __init__(self, device: Optional[Device] = None) -> None:
        """
        :param device:
            If ``None``; if CUDA is available, the process will use the first
            CUDA device; otherwise, it will use the CPU.
        """
        if device is None:
            device = _determine_default_device()

        super().__init__(rank=0, size=1, device=device)

    @override
    def close(self) -> None:
        pass

    @override
    def as_process_group(self) -> ProcessGroup:
        raise RuntimeError("`FakeGang` does not support conversion to a process group.")

    @override
    def barrier(self) -> None:
        pass

    @override
    def all_reduce(self, tensor: Tensor, op: ReduceOperation) -> None:
        pass

    @override
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor) -> None:
        output_tensor.copy_(input_tensor)


@final
class ProcessGroupGang(AbstractGang):
    """Represents a gang that wraps a process group."""

    _pg: ProcessGroup

    def __init__(self, pg: ProcessGroup, device: Device) -> None:
        super().__init__(dist.get_rank(pg), dist.get_world_size(pg), device)

        self._pg = pg

    @staticmethod
    def init_default_process_group(
        *,
        device: Optional[Device] = None,
        timeout: Optional[timedelta] = None,
        num_threads: Optional[int] = None,
        warn_only: bool = False,
        ok_initialized: bool = False,
    ) -> Gang:
        """Initialize the default process group and wrap it as a gang.

        :param device:
            If ``None``; if CUDA is available, the process group will be
            initialized on an automatically selected CUDA device; otherwise,
            it will be initialized on the CPU.
        :param timeout:
            The timeout for operations executed against the process group.
        :param num_threads:
            The number of threads used for interaop parallelism.
        :param warn_only:
            If ``True``, logs a warning instead of raising an error if the
            process group is not set up reliably.
        :param ok_initialized:
            If ``True``, does not raise an error if the process group is already
            initialized.
        """
        if not dist.is_available():
            raise RuntimeError("`torch.distributed` is not available.")

        if dist.is_initialized():
            if ok_initialized:
                return ProcessGroupGang.from_default_process_group()

            raise RuntimeError("The default process group is already initialized.")

        num_procs = get_local_world_size()

        if num_threads is None:
            if num_procs > 1 and "OMP_NUM_THREADS" not in os.environ:
                # To prevent thread oversubscription, we distribute cores evenly
                # across the workers.
                num_threads = _get_num_cpus(num_procs)

        if num_threads is not None:
            torch.set_num_threads(num_threads)

            log.info("Setting the number of threads used for intraop parallelism to {}.", num_threads)  # fmt: skip

        if device is None:
            device = _determine_default_device()

            assert device.type == "cpu" or device.type == "cuda"

        if device.type == "cpu":
            backend = "gloo"
        elif device.type == "cuda":
            backend = "nccl"
        else:
            raise ValueError(
                f"`device` must be of type `cpu` and `cuda`, but is of type `{device.type}` instead."
            )

        if device.type == "cuda":

            def check_async_handling() -> None:
                env_name = "NCCL_ASYNC_ERROR_HANDLING"
                if env_name in os.environ:
                    return

                if _is_pt22_or_greater():
                    env_name = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
                    if env_name in os.environ:
                        return

                if warn_only:
                    log.warning("The default process group uses the `nccl` backend, but the `{}` environment variable is not set. Your collective communication calls can hang indefinitely. Learn more at https://github.com/pytorch/pytorch/issues/46874.", env_name)  # fmt: skip
                else:
                    raise RuntimeError(
                        f"The default process group uses the `nccl` backend, but the `{env_name}` environment variable is not set. Learn more at https://github.com/pytorch/pytorch/issues/46874."
                    )

            check_async_handling()

        if timeout is None:
            timeout = timedelta(minutes=15)

        dist.init_process_group(backend, timeout=timeout)

        if dist.group.WORLD is None:
            raise RuntimeError(
                "The default process group is not available. Please file a bug report."
            )

        return ProcessGroupGang(dist.group.WORLD, device)

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
        """Wrap the default process group as a gang."""
        if not dist.is_available():
            raise RuntimeError("`torch.distributed` is not available.")

        if not dist.is_initialized():
            raise RuntimeError("The default process group is not initialized.")

        backend = dist.get_backend()

        if backend == "gloo":
            device = CPU
        elif backend == "nccl":
            device = _determine_default_cuda_device()
        else:
            raise RuntimeError(
                f"Only `nccl` and `gloo` backends are supported, but the process group uses the `{backend}` backend."
            )

        if dist.group.WORLD is None:
            raise RuntimeError(
                "The default process group is not available. Please file a bug report."
            )

        return ProcessGroupGang(dist.group.WORLD, device)

    @override
    def close(self) -> None:
        dist.destroy_process_group(self._pg)

    @override
    def as_process_group(self) -> ProcessGroup:
        return self._pg

    @override
    def barrier(self) -> None:
        dist.barrier(group=self._pg)

    @override
    def all_reduce(self, tensor: Tensor, op: ReduceOperation) -> None:
        dist.all_reduce(tensor, self._get_reduce_op(op), group=self._pg)

    @override
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor) -> None:
        dist.all_gather_into_tensor(output_tensor, input_tensor, group=self._pg)

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


def _get_num_cpus(num_procs: int) -> int:
    num_cpus = os.cpu_count()
    if num_cpus is None:
        log.warning("The number of CPU cores cannot be determined.")

        return 1

    max_num_cpus = max(num_cpus // num_procs, 1)

    # We should not exceed the number of cores available in the affinity mask.
    return min(max_num_cpus, len(os.sched_getaffinity(0)))


_default_device: Optional[Device] = None


def _determine_default_device() -> Device:
    global _default_device

    if _default_device is not None:
        return _default_device

    device_str = os.getenv("FAIRSEQ2_DEVICE")
    if device_str is not None:
        try:
            _default_device = Device(device_str)
        except RuntimeError as ex:
            raise RuntimeError(
                f"The value of the `FAIRSEQ2_DEVICE` environment variable must specify a valid PyTorch device, but is '{device_str}' instead."
            ) from ex

    if _default_device is None:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            _default_device = _determine_default_cuda_device()

    if _default_device is None:
        _default_device = CPU

    if _default_device.type == "cuda":
        torch.cuda.set_device(_default_device)

    log.info("Setting '{}' as the default device of the process.", _default_device)

    return _default_device


def _determine_default_cuda_device() -> Device:
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        try:
            int(visible_devices)
        except ValueError:
            # If we are here, it means CUDA_VISIBLE_DEVICES is a list instead of
            # a single device index.
            device = None
        else:
            device = Device("cuda", index=0)
    else:
        device = None

    if device is None:
        num_devices = torch.cuda.device_count()

        idx = _get_device_index(num_devices, device_type="cuda")

        device = Device("cuda", index=idx)

    return device


def _get_device_index(num_devices: int, device_type: str) -> int:
    assert num_devices > 0

    # We use the `LOCAL_RANK` environment variable to determine which device to
    # pick in case the process has more than one available.
    device_idx = _get_int_from_env("LOCAL_RANK", allow_zero=True)
    if device_idx is None:
        num_procs = get_local_world_size()
        if num_procs > 1 and num_devices > 1:
            raise RuntimeError(
                f"The default `{device_type}` device cannot be determined. There are {num_devices} devices available, but the `LOCAL_RANK` environment variable is not set."
            )

        return 0

    if device_idx < 0:
        raise RuntimeError(
            f"The value of the `LOCAL_RANK` environment variable must be greater than or equal to 0, but is {device_idx} instead."
        )

    if device_idx >= num_devices:
        raise RuntimeError(
            f"The value of the `LOCAL_RANK` environment variable must be less than the number of available `{device_type}` devices ({num_devices}), but is {device_idx} instead."
        )

    return device_idx


def get_world_size() -> int:
    """Return the world size of the running job."""
    value = _get_int_from_env("WORLD_SIZE")

    return 1 if value is None else value


def get_rank() -> int:
    """Return the rank of this process in the running job."""
    value = _get_int_from_env("RANK", allow_zero=True)

    return 0 if value is None else value


def get_local_world_size() -> int:
    """Return the local world size of the running job."""
    value = _get_int_from_env("LOCAL_WORLD_SIZE")

    return 1 if value is None else value


def get_local_rank() -> int:
    """Return the local rank of this process in the running job."""
    value = _get_int_from_env("LOCAL_RANK", allow_zero=True)

    return 0 if value is None else value


def _get_int_from_env(var_name: str, allow_zero: bool = False) -> Optional[int]:
    s = os.getenv(var_name)
    if s is None:
        return None

    try:
        value = int(s)
    except ValueError:
        raise RuntimeError(
            f"The value of the `{var_name}` environment variable must be an integer, but is '{value}' instead."
        )

    if not allow_zero:
        if not value >= 1:
            raise RuntimeError(
                f"The value of the `{var_name}` environment variable must be greater than 0, but is {value} instead."
            )
    else:
        if not value >= 0:
            raise RuntimeError(
                f"The value of the `{var_name}` environment variable must be greater than or equal to 0, but is {value} instead."
            )

    return value
