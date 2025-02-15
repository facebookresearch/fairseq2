# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Any, final

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import Backend, ProcessGroup, ReduceOp
from typing_extensions import override

from fairseq2.error import InternalError, InvalidOperationError, NotSupportedError
from fairseq2.logging import log
from fairseq2.typing import CPU, Device
from fairseq2.utils.env import (
    InvalidEnvironmentVariableError,
    get_local_world_size,
    get_world_size,
)
from fairseq2.utils.version import torch_greater_or_equal


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
        """Close and destroy the gang."""

    @abstractmethod
    def create_gang(self, ranks: Sequence[int]) -> Gang | None:
        """Make a new gang.

        :param ranks:
            The ranks of processes that will be part of the new gang.
        """

    @abstractmethod
    def as_process_group(self) -> ProcessGroup:
        """Return this gang as a process group."""

    @abstractmethod
    def barrier(self) -> None:
        """Synchronize all processes."""

    @abstractmethod
    def all_reduce(self, tensor: Tensor, op: ReduceOperation) -> None:
        """Reduce ``tensor`` across all processes.

        :param tensor:
            The input and output tensor of the operation.
        :param op:
            The element-wise reduce operation.
        """

    @abstractmethod
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor) -> None:
        """Gather tensors from all processes and put them in ``output_tensor``.

        :param output_tensor:
            The output tensor to accomodate tensors from all processes.
        :param input_tensor:
            The tensor to be gathered from this process.
        """

    @abstractmethod
    def all_gather_to_list(
        self, output_tensors: list[Tensor], input_tensor: Tensor
    ) -> None:
        """Gather tensors from all processes and put them in ``output_tensors``.

        :param output_tensors:
            The tensor list to accomodate tensors from all processes.
        :param input_tensor:
            The tensor to be gathered from this process.
        """

    @abstractmethod
    def broadcast(self, tensor: Tensor, source_rank: int = 0) -> None:
        """Broadcast ``tensor`` from ``source_rank`` to all processes.

        :param tensor:
            The tensor to be sent from ``source_rank``.
        :param source_rank:
            The rank of the process from which to broadcast ``tensor``.
        """

    @abstractmethod
    def broadcast_objects(self, objects: list[object], source_rank: int = 0) -> None:
        """Broadcast picklable ``objects`` from ``source_rank`` to all processes.

        :param objects:
            The list of picklable objects to broadcast. Each process must
            provide lists of equal sizes.
        :param source_rank:
            The rank of the process from which to broadcast ``objects``.
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


class GangError(Exception):
    pass


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
        if size == 0:
            raise ValueError("`size` must be greater than zero.")

        if rank >= size:
            raise ValueError(
                f"`rank` must be less than `size` ({size}), but is {rank} instead."
            )

        if device.type == "meta":
            raise ValueError("`device` must be a real device.")

        self._rank = rank
        self._size = size

        self._device = device

    @final
    @override
    def create_gang(self, ranks: Sequence[int]) -> Gang | None:
        if len(set(ranks)) != len(ranks):
            raise ValueError("The ranks in ``ranks`` must be all unique.")

        for idx, rank in enumerate(ranks):
            if rank < 0 or rank > self._size:
                raise ValueError(
                    f"The rank at index {idx} in ``ranks`` must be greater than or equal to 0 and less than the size of the gang ({self._size}), but is {rank} instead."
                )

        return self._do_create_gang(ranks)

    @abstractmethod
    def _do_create_gang(self, ranks: Sequence[int]) -> Gang | None:
        """Make a new gang.

        :param ranks:
            The ranks of processes that will be part of the new gang.
        """

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

    def __init__(
        self, device: Device | None = None, *, rank: int = 0, size: int = 1
    ) -> None:
        """
        :param device: If ``None``, CPU will be used.
        :param rank: The emulated rank of this process in the gang.
        :param size: The emulated number of processes that are part of the gang.
        """
        super().__init__(rank=rank, size=size, device=device or CPU)

    @override
    def close(self) -> None:
        pass

    @override
    def _do_create_gang(self, ranks: Sequence[int]) -> FakeGang | None:
        try:
            idx = ranks.index(self._rank)
        except ValueError:
            return None

        return FakeGang(rank=idx, size=len(ranks), device=self._device)

    @override
    def as_process_group(self) -> ProcessGroup:
        raise NotSupportedError(
            "`FakeGang` does not support conversion to a process group."
        )

    @override
    def barrier(self) -> None:
        pass

    @override
    def all_reduce(self, tensor: Tensor, op: ReduceOperation) -> None:
        match op:
            case ReduceOperation.SUM:
                tensor *= self._size
            case ReduceOperation.PRODUCT:
                tensor.pow_(self._size)
            case _:
                raise NotSupportedError(
                    "`FakeGang` supports only `SUM` and `PRODUCT` reduce operations."
                )

    @override
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor) -> None:
        if not output_tensor.is_contiguous():
            raise ValueError("`output_tensor` must be contiguous.")

        if output_tensor.dim() != input_tensor.dim() + 1:
            raise ValueError(
                "`output_tensor` must have a shape that is compatible with all-gather."
            )

        if output_tensor.size(0) != self._size:
            raise ValueError(
                f"The size of the first dimension of `output_tensor` must match the number of processes in the gang ({self._size}), but is {output_tensor.size(0)} instead."
            )

        for i in range(self._size):
            output_tensor[i].copy_(input_tensor)

    @override
    def all_gather_to_list(
        self, output_tensors: list[Tensor], input_tensor: Tensor
    ) -> None:
        if len(output_tensors) != self._size:
            raise ValueError(
                f"The length of `output_tensors` must match the number of processes in the gang ({self._size}), but is {len(output_tensors)} instead."
            )

        for i in range(self._size):
            output_tensors[i].copy_(input_tensor)

    @override
    def broadcast(self, tensor: Tensor, source_rank: int = 0) -> None:
        if source_rank != self._rank:
            raise ValueError(
                f"`source_rank` must be {self._rank}, but is {source_rank} instead."
            )

    @override
    def broadcast_objects(self, objects: list[object], source_rank: int = 0) -> None:
        if source_rank != self._rank:
            raise ValueError(
                f"`source_rank` must be {self._rank}, but is {source_rank} instead."
            )


@final
class ProcessGroupGang(AbstractGang):
    """Represents a gang that wraps a process group."""

    _pg: ProcessGroup
    _monitor_pg: ProcessGroup | None

    def __init__(
        self,
        pg: ProcessGroup,
        device: Device,
        *,
        monitor_pg: ProcessGroup | None = None,
    ) -> None:
        super().__init__(dist.get_rank(pg), dist.get_world_size(pg), device)

        self._pg = pg
        self._monitor_pg = monitor_pg

    @classmethod
    def init_root_process_group(
        cls,
        device: Device,
        *,
        timeout: timedelta | None = None,
        high_priority: bool = False,
        num_threads: int | None = None,
        monitored: bool = False,
    ) -> ProcessGroupGang:
        """Initialize the root process group and wrap it as a gang.

        :param device: The device for which to initialize the gang. For CUDA
            devices, NCCL; for CPU, Gloo will be used.
        :param timeout: The timeout for collective operations. If ``None``, the
            default timeout value (15 minutes) will be used.
        :param num_threads: The number of threads to use for interaop
            parallelism.
        :param high_priority: If ``True``, the underlying collective operations
            will be performed on high priority channels (e.g. CUDA streams).
        :param monitored: If ``True``,  puts a monitored barrier before every
            collective call for troubleshooting purposes.
        """
        if log.is_enabled_for_debug():
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

            dist.set_debug_level_from_env()

        if not dist.is_available():
            raise GangError("`torch.distributed` is not available.")

        if dist.is_initialized():
            raise GangError("The default process group is already initialized.")

        backend: str | None

        if device.type == "cpu":
            backend = Backend.GLOO
        elif device.type == "cuda":
            backend = Backend.NCCL
        else:
            raise NotSupportedError(
                f"`device` must be of type `cpu` and `cuda`, but is of type `{device.type}` instead."
            )

        if num_threads is None:
            try:
                num_procs = get_local_world_size(os.environ)
            except InvalidEnvironmentVariableError as ex:
                raise GangError(
                    "The local world size cannot be determined from the environment variables. See the nested exception for details."
                ) from ex

            if num_procs > 1 and "OMP_NUM_THREADS" not in os.environ:
                # To prevent thread oversubscription, we distribute cores evenly
                # across the workers.
                num_threads = _get_num_cpus(num_procs)

        if num_threads is not None:
            torch.set_num_threads(num_threads)

            log.info("Setting the number of threads used for intra-op parallelism to {}.", num_threads)  # fmt: skip

        if device.type == "cuda":
            # See https://github.com/pytorch/pytorch/issues/46874.
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

        if timeout is None:
            timeout = timedelta(minutes=15)

        kwargs: dict[str, Any] = {}

        if torch_greater_or_equal(2, 3):
            # Forces NCCL to initialize immediately which enables deterministic
            # behavior.
            kwargs = {"device_id": device}

        # If enabled, uses high priority CUDA streams for NCCL.
        if device.type == "cuda" and high_priority:
            # Not available unless PyTorch is built with NCCL.
            from torch.distributed import ProcessGroupNCCL

            pg_options = ProcessGroupNCCL.Options(is_high_priority_stream=True)
        else:
            pg_options = None

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore", message=".*options._timeout was specified.*"
                )

                dist.init_process_group(
                    backend, timeout=timeout, pg_options=pg_options, **kwargs
                )
        except (RuntimeError, ValueError) as ex:
            raise GangError(
                "The underlying process group has failed to initialize. See the nested exception for details."
            ) from ex

        pg = dist.group.WORLD
        if pg is None:
            raise InternalError("`dist.group.WORLD` is `None`.")

        if monitored:
            if backend == Backend.GLOO:
                monitor_pg = pg
            else:
                # Gloo is needed for monitored barrier support.
                try:
                    monitor_pg = dist.new_group(backend=Backend.GLOO, timeout=timeout)
                except RuntimeError as ex:
                    raise GangError(
                        "The underlying process group used for monitoring has failed to initialize. See the nested exception for details."
                    ) from ex
        else:
            monitor_pg = None

        return ProcessGroupGang(pg, device, monitor_pg=monitor_pg)

    @override
    def close(self) -> None:
        dist.destroy_process_group(self._pg)

    @override
    def _do_create_gang(self, ranks: Sequence[int]) -> ProcessGroupGang | None:
        if self._pg is not dist.group.WORLD:
            raise InvalidOperationError(
                "`create_gang()` can only be called on the gang associated with the default (i.e. main) process group."
            )

        try:
            backend = dist.get_backend()
        except RuntimeError as ex:
            raise GangError(
                "The default process group backend cannot be determined. See the nested exception for details."
            ) from ex

        try:
            pg = dist.new_group(ranks, backend=backend)
        except RuntimeError as ex:
            s = ", ".join(sorted(str(r) for r in ranks))

            raise GangError(
                f"The creation of a new child process group has failed for ranks {s}. See the nested exception for details."
            ) from ex

        if self._rank not in ranks:
            return None

        if self._monitor_pg is not None:
            if backend == Backend.GLOO:
                monitor_pg = pg
            else:
                try:
                    monitor_pg = dist.new_group(ranks, backend=Backend.GLOO)
                except RuntimeError as ex:
                    s = ", ".join(sorted(str(r) for r in ranks))

                    raise GangError(
                        f"The creation of a new monitoring child process group has failed for ranks {s}. See the nested exception for details."
                    ) from ex
        else:
            monitor_pg = None

        return ProcessGroupGang(pg, self._device, monitor_pg=monitor_pg)

    @override
    def as_process_group(self) -> ProcessGroup:
        return self._pg

    @override
    def barrier(self) -> None:
        if self._monitor_pg is None:
            try:
                dist.barrier(group=self._pg, device_ids=[self._device.index])
            except RuntimeError as ex:
                raise GangError(
                    "The `barrier` collective operation has failed. See the nested exception for details."
                ) from ex
        else:
            torch.cuda.synchronize()

            try:
                dist.monitored_barrier(group=self._monitor_pg, wait_all_ranks=True)
            except RuntimeError as ex:
                raise GangError(
                    "The `monitored_barrier` collective operation has failed. See the nested exception for details."
                ) from ex

    @override
    def all_reduce(self, tensor: Tensor, op: ReduceOperation) -> None:
        self._maybe_monitored_barrier()

        try:
            dist.all_reduce(tensor, self._get_reduce_op(op), group=self._pg)
        except RuntimeError as ex:
            raise GangError(
                "The `all_reduce` collective operation has failed. See the nested exception for details."
            ) from ex

    @override
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor) -> None:
        self._maybe_monitored_barrier()

        try:
            dist.all_gather_into_tensor(output_tensor, input_tensor, group=self._pg)
        except RuntimeError as ex:
            raise GangError(
                "The `all_gather` collective operation has failed. See the nested exception for details."
            ) from ex

    @override
    def all_gather_to_list(
        self, output_tensors: list[Tensor], input_tensor: Tensor
    ) -> None:
        self._maybe_monitored_barrier()

        try:
            dist.all_gather(output_tensors, input_tensor, group=self._pg)
        except RuntimeError as ex:
            raise GangError(
                "The `all_gather_to_list` collective operation has failed. See the nested exception for details."
            ) from ex

    @override
    def broadcast(self, tensor: Tensor, source_rank: int = 0) -> None:
        self._maybe_monitored_barrier()

        try:
            dist.broadcast(tensor, source_rank, group=self._pg)
        except RuntimeError as ex:
            raise GangError(
                "The `broadcast` collective operation has failed. See the nested exception for details."
            ) from ex

    @override
    def broadcast_objects(self, objects: list[object], source_rank: int = 0) -> None:
        self._maybe_monitored_barrier()

        try:
            dist.broadcast_object_list(objects, source_rank, group=self._pg)
        except RuntimeError as ex:
            raise GangError(
                "The `broadcast_object_list` collective operation has failed. See the nested exception for details."
            ) from ex

    def _maybe_monitored_barrier(self) -> None:
        if self._monitor_pg is None:
            return

        torch.cuda.synchronize()

        try:
            dist.monitored_barrier(group=self._monitor_pg, wait_all_ranks=True)
        except RuntimeError as ex:
            raise GangError(
                "The `monitored_barrier` collective operation has failed. See the nested exception for details."
            ) from ex

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

        raise NotSupportedError(
            f"`{op}` operation is not supported by the underlying process group."
        )


def _get_num_cpus(num_procs: int) -> int:
    num_cpus = os.cpu_count()

    affinity_mask = os.sched_getaffinity(0)

    if num_cpus is None or affinity_mask is None:
        log.warning("The number of CPU cores cannot be determined.")

        return 1

    # We should not exceed the number of cores available in the affinity mask.
    return min(max(num_cpus // num_procs, 1), len(affinity_mask))


def setup_root_gang(
    device: Device,
    *,
    timeout: timedelta | None = None,
    high_priority: bool = False,
    monitored: bool = False,
) -> Gang:
    """Create the root gang of this process.

    :param device: The device for which to initialize the gang. For CUDA
        devices, NCCL; for CPU, Gloo will be used.
    :param timeout: The timeout for collective operations. If ``None``, the
        default timeout value (15 minutes) will be used.
    :param high_priority: If ``True``, the underlying collective operations
        will be performed on high priority channels (e.g. CUDA streams).
    :param monitored: If ``True``,  puts a monitored barrier before every
        collective call for troubleshooting purposes.
    """
    try:
        world_size = get_world_size(os.environ)
    except InvalidEnvironmentVariableError as ex:
        raise GangError(
            "The world size cannot be determined. See the nested exception for details."
        ) from ex

    if world_size == 1:
        return FakeGang(device)

    return ProcessGroupGang.init_root_process_group(
        device, timeout=timeout, high_priority=high_priority, monitored=monitored
    )


@dataclass
class Gangs:
    root: Gang
    dp: Gang
    tp: Gang

    def close(self) -> None:
        self.root.close()


def fake_gangs(device: Device) -> Gangs:
    gang = FakeGang(device=device)

    return Gangs(gang, gang, gang)


def to_gangs(gang: Gang) -> Gangs:
    fake_gang = FakeGang(device=gang.device)

    return Gangs(gang, gang, fake_gang)


def setup_parallel_gangs(root_gang: Gang, *, tp_size: int = 1) -> Gangs:
    """Sets up gangs to be used for data and model parallelism.

    For instance; if we have 8 devices denoted by g0 to g7 and 2 devices are
    used for tensor parallelism, this function will make 4 tensor parallel
    gangs and 2 data parallel gangs as:

        4 tensor parallel gangs:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 data parallel gangs:
            [g0, g2, g4, g6], [g1, g3, g5, g7]

    For efficiency, the caller should make sure adjacent ranks are on the same
    host. For example, if there are two hosts with a total of 16 GPUs, ranks 0
    to 7 belong to the first host and ranks 8 to 15 belong to the second host.

    :param root_gang: The gang whose topology will be used to make the new gangs.
    :param tp_size: The size of tensor parallel gangs.
    """
    if tp_size < 1:
        raise ValueError(f"`tp_size` must be greater than 0, but is {tp_size} instead.")

    if tp_size > root_gang.size:
        raise ValueError(
            f"`tp_size` must be less than or equal to the number of processes in the root gang ({root_gang.size}), but is {tp_size} instead."
        )

    if root_gang.size % tp_size != 0:
        raise ValueError(
            f"`root_gang.size` is expected to be a multiple of `tp_size` ({tp_size}), but is {root_gang.size} instead."
        )

    dp_size = root_gang.size // tp_size

    mesh = torch.arange(root_gang.size).view(dp_size, tp_size)

    # Get the coordinate of this process in the mesh.
    rank_coords = [x.item() for x in torch.where(mesh == root_gang.rank)]

    dp_gang: Gang | None = None

    log.info("Initializing data parallel gang with a size of {}.", dp_size)

    # Build the gangs for data parallelism.
    match dp_size:
        case 1:
            dp_gang = FakeGang(device=root_gang.device)
        case root_gang.size:
            dp_gang = root_gang
        case _:
            for i in range(tp_size):
                sub_gang = root_gang.create_gang(mesh[:, i].tolist())
                if i == rank_coords[1]:
                    dp_gang = sub_gang

    if dp_gang is None:
        raise InternalError("`dp_gang` is `None`.")

    tp_gang: Gang | None = None

    log.info("Initializing tensor parallel gang with a size of {}.", tp_size)

    # Build the gangs for tensor parallelism.
    match tp_size:
        case 1:
            tp_gang = FakeGang(device=root_gang.device)
        case root_gang.size:
            tp_gang = root_gang
        case _:
            for i in range(dp_size):
                sub_gang = root_gang.create_gang(mesh[i, :].tolist())
                if i == rank_coords[0]:
                    tp_gang = sub_gang

    if tp_gang is None:
        raise InternalError("`tp_gang` is `None`.")

    return Gangs(root_gang, dp_gang, tp_gang)


def setup_hsdp_gangs(dp_gang: Gang, local_world_size: int) -> tuple[Gang, Gang]:
    """
    Sets up gangs to be used for hybrid sharded data parallelism.

    For instance; if we have 8 devices denoted by g0 to g7 and ``local_world_size``
    is 4, this function will make 2 intra-node gangs and 4 inter-node gangs:

        2 intra-node gangs of size 4:
            [g0, g1, g2, g3], [g4, g5, g6, g7]
        4 inter-node gangs of size 2:
            [g0, g4], [g1, g5], [g2, g6], [g3, g7]

    For efficiency, the caller should make sure adjacent ranks are on the same
    host.

    :returs:
        A tuple of intra-node gang for sharding and inter-node gang for
        replication.
    """
    if local_world_size <= 1:
        raise ValueError(
            f"`local_world_size` must be greater than 1, but is {local_world_size} instead."
        )

    if dp_gang.size % local_world_size != 0:
        raise ValueError(
            f"`dp_gang.size` is expected to be a multiple of `local_world_size` ({local_world_size}), but is {dp_gang.size} instead."
        )

    intra_node_size = local_world_size

    inter_node_size = dp_gang.size // local_world_size

    mesh = torch.arange(dp_gang.size).view(inter_node_size, intra_node_size)

    # Get the coordinate of this process in the mesh.
    rank_coords = [x.item() for x in torch.where(mesh == dp_gang.rank)]

    inter_node_gang: Gang | None = None

    log.info("Initializing inter-node data parallel gang with a size of {}.", inter_node_size)  # fmt: skip

    # Build the gangs for inter-node data parallelism.
    match inter_node_size:
        case 1:
            inter_node_gang = FakeGang(device=dp_gang.device)
        case dp_gang.size:
            inter_node_gang = dp_gang
        case _:
            for i in range(intra_node_size):
                sub_gang = dp_gang.create_gang(mesh[:, i].tolist())
                if i == rank_coords[1]:
                    inter_node_gang = sub_gang

    if inter_node_gang is None:
        raise InternalError("`inter_node_gang` is `None`.")

    intra_node_gang: Gang | None = None

    log.info("Initializing intra-node data parallel gang with a size of {}.", intra_node_size)  # fmt: skip

    # Build the gangs for intra-node data parallelism.
    match intra_node_size:
        case 1:
            intra_node_gang = FakeGang(device=dp_gang.device)
        case dp_gang.size:
            intra_node_gang = dp_gang
        case _:
            for i in range(inter_node_size):
                sub_gang = dp_gang.create_gang(mesh[i, :].tolist())
                if i == rank_coords[0]:
                    intra_node_gang = sub_gang

    if intra_node_gang is None:
        raise InternalError("`intra_node_gang` is `None`.")

    return intra_node_gang, inter_node_gang


def broadcast_flag(gang: Gang, flag: bool, source_rank: int = 0) -> bool:
    """Broadcast ``flag`` to  all processes in ``gang`` from ``source_rank``."""
    tmp = torch.tensor(flag, device=gang.device)

    gang.broadcast(tmp, source_rank)

    return bool(tmp)


def all_sum(gang: Gang, value: float | int | Tensor) -> Tensor:
    """Sum ``value`` over all processes in ``gang``."""
    if isinstance(value, Tensor):
        output = value
    else:
        output = torch.tensor(value, device=gang.device)

    gang.all_reduce(output, ReduceOperation.SUM)

    return output
