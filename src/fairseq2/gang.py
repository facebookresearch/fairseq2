# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import final

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import Backend, ProcessGroup, ReduceOp
from typing_extensions import override

from fairseq2.device import determine_default_cuda_device, determine_default_device
from fairseq2.error import InternalError, InvalidOperationError, NotSupportedError
from fairseq2.logging import log
from fairseq2.typing import CPU, Device
from fairseq2.utils.env import InvalidEnvironmentVariableError, get_int_from_env


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
    def make_gang(self, ranks: Sequence[int]) -> Gang | None:
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

        self._rank = rank
        self._size = size

        self._device = device

    @final
    @override
    def make_gang(self, ranks: Sequence[int]) -> Gang | None:
        if len(set(ranks)) != len(ranks):
            raise ValueError("The ranks in ``ranks`` must be all unique.")

        for idx, rank in enumerate(ranks):
            if rank < 0 or rank > self._size:
                raise ValueError(
                    f"The rank at index {idx} in ``ranks`` must be greater than or equal to 0 and less than the size of the gang ({self._size}), but is {rank} instead."
                )

        return self._do_make_gang(ranks)

    @abstractmethod
    def _do_make_gang(self, ranks: Sequence[int]) -> Gang | None:
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
        self, *, rank: int = 0, size: int = 1, device: Device | None = None
    ) -> None:
        """
        :param rank:
            The emulated rank of this process in the gang.
        :param size:
            The emulated number of processes that are part of the gang.
        :param device:
            If ``None``; if CUDA is available, the gang will use the default
            CUDA device of the process; otherwise, it will use the CPU.
        """
        if device is None:
            device = determine_default_device()

        super().__init__(rank=rank, size=size, device=device)

    @override
    def close(self) -> None:
        pass

    @override
    def _do_make_gang(self, ranks: Sequence[int]) -> FakeGang | None:
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

    _default: ProcessGroupGang | None = None

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
    def init_default_process_group(
        cls,
        *,
        device: Device | None = None,
        timeout: timedelta | None = None,
        num_threads: int | None = None,
        monitored: bool = False,
        ok_initialized: bool = False,
    ) -> ProcessGroupGang:
        """Initialize the default process group and wrap it as a gang.

        :param device:
            If ``None``; if CUDA is available, the gang will use the default
            CUDA device of the process; otherwise, it will use the CPU.
        :param timeout:
            The timeout for collective operations. If ``None``, the default
            timeout value (15 minutes) will be used.
        :param num_threads:
            The number of threads to use for interaop parallelism.
        :param monitored:
            If ``True``,  puts a monitored barrier before every collective call.
        :param ok_initialized:
            If ``True``, does not raise an error if the default process group is
            already initialized.
        """
        if log.is_enabled_for_debug():
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

            dist.set_debug_level_from_env()

        if not dist.is_available():
            raise GangError("`torch.distributed` is not available.")

        if dist.is_initialized():
            if ok_initialized:
                log.info("Default process group is already initialized. Skipping initialization.")  # fmt: skip

                return ProcessGroupGang.from_default_process_group()

            raise GangError("The default process group is already initialized.")

        try:
            num_procs = get_local_world_size()
        except InvalidEnvironmentVariableError as ex:
            raise GangError(
                "The local world size cannot be determined from the environment variables. See the nested exception for details."
            ) from ex

        if num_threads is None:
            if num_procs > 1 and "OMP_NUM_THREADS" not in os.environ:
                # To prevent thread oversubscription, we distribute cores evenly
                # across the workers.
                num_threads = _get_num_cpus(num_procs)

        if num_threads is not None:
            torch.set_num_threads(num_threads)

            log.info("Setting the number of threads used for intraop parallelism to {}.", num_threads)  # fmt: skip

        if device is None:
            device = determine_default_device()

            if device.type != "cpu" and device.type != "cuda":
                raise InternalError(f"`device` is `{device}`.")

        backend: str | None

        if device.type == "cpu":
            backend = Backend.GLOO
        elif device.type == "cuda":
            backend = Backend.NCCL
        else:
            raise ValueError(
                f"`device` must be of type `cpu` and `cuda`, but is of type `{device.type}` instead."
            )

        if device.type == "cuda":
            # See https://github.com/pytorch/pytorch/issues/46874.
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

        if timeout is None:
            timeout = timedelta(minutes=15)

        try:
            dist.init_process_group(backend, timeout=timeout)
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

        cls._default = ProcessGroupGang(pg, device, monitor_pg=monitor_pg)

        return cls._default

    @staticmethod
    def from_process_group(pg: ProcessGroup, device: Device) -> ProcessGroupGang:
        """Wrap ``pg`` as a gang.

        :param pg:
            The process group to wrap.
        :param device:
            The associated device.
        """
        return ProcessGroupGang(pg, device)

    @classmethod
    def from_default_process_group(cls) -> ProcessGroupGang:
        """Wrap the default process group as a gang."""
        if not dist.is_available():
            raise GangError("`torch.distributed` is not available.")

        if not dist.is_initialized():
            raise GangError("The default process group is not initialized.")

        if cls._default is not None:
            return cls._default

        try:
            backend = dist.get_backend()
        except RuntimeError as ex:
            raise GangError(
                "The default process group backend cannot be determined. See the nested exception for details."
            ) from ex

        match backend:
            case Backend.GLOO:
                device = CPU
            case Backend.NCCL:
                cuda_device = determine_default_cuda_device()
                if cuda_device is None:
                    raise GangError(
                        "The default process group uses the `nccl` backend, but the `cuda` device cannot be determined."
                    )

                device = cuda_device
            case _:
                raise NotSupportedError(
                    f"Only `nccl` and `gloo` backends are supported, but the process group uses the `{backend}` backend."
                )

        if dist.group.WORLD is None:
            raise InternalError("`dist.group.WORLD` is `None`.")

        cls._default = ProcessGroupGang(dist.group.WORLD, device)

        return cls._default

    @override
    def close(self) -> None:
        dist.destroy_process_group(self._pg)

    @override
    def _do_make_gang(self, ranks: Sequence[int]) -> ProcessGroupGang | None:
        if self._pg is not dist.group.WORLD:
            raise InvalidOperationError(
                "`make_gang()` can only be called on the gang associated with the default (i.e. main) process group."
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
        log.warning("The number of CPUs cannot be determined.")

        return 1

    # We should not exceed the number of cores available in the affinity mask.
    return min(max(num_cpus // num_procs, 1), len(affinity_mask))


def setup_default_gang(
    *,
    device: Device | None = None,
    timeout: timedelta | None = None,
    monitored: bool = False,
) -> Gang:
    """Make the default gang of this process.

    :param device:
        If ``None``; if CUDA is available, the gang will use the default CUDA
        device of the process; otherwise, it will use the CPU.
    :param timeout:
        The timeout for collective operations.
    :param monitored:
        If ``True``,  puts a monitored barrier before every collective call.
    """
    try:
        world_size = get_world_size()
    except InvalidEnvironmentVariableError as ex:
        raise GangError(
            "The world size cannot be determined from the environment variables. See the nested exception for details."
        ) from ex

    if world_size == 1:
        return FakeGang(device=device)

    return ProcessGroupGang.init_default_process_group(
        device=device, timeout=timeout, monitored=monitored, ok_initialized=True
    )


@dataclass
class Gangs:
    root: Gang
    dp: Gang
    tp: Gang


def fake_gangs(device: Device) -> Gangs:
    gang = FakeGang(device=device)

    return Gangs(gang, gang, gang)


def _setup_2D_mesh_gangs(
    root_gang: Gang,
    *,
    row_length: int = 1,
    create_single_rank_process_groups: bool = False,
    dim_descriptions: list[str] | None = None,
) -> dict[int, Gang]:
    """Set up gangs for this process as defined by a 2D device mesh.

    The two returned gangs are defined by the process' position in the mesh.
    First gang is the row in the mesh, second is the column.
    For example, assuming 8 devices denoted by g0 to g7, calling this function
    with ``row_length`` = 4 amounts to defining the 2D mesh
    [[g0, g1, g2, g3], [g4, g5, g6, g7]] and making 2 sets of gangs:

        2 gangs of size 4 (mesh rows):
            [g0, g1, g2, g3], [g4, g5, g6, g7]
        4 gangs of size 2 (mesh columns):
            [g0, g4], [g1, g5], [g2, g6], [g3, g7]

    For the process of rank 5, the function would return the 2 sub-gangs
    {0: [g4, g5, g6, g7], 1: [g1, g5]}. If adjacent ranks are on the same host
    (for example, 2 hosts: one with g0 to g3, and the other with g4 to g7),
    the first gang can be used to maximize local intra-host communication.

    Example use-cases include making tensor- and data- parallel gangs, or
    sharding and replicating gangs in FSDP's hybrid sharding.

    :param root_gang:
        The gang whose topology will be used to make the new gangs.
    :param row_length:
        The size of the gangs corresponding to the 2D mesh rows.
    :param create_single_rank_process_groups:
        If ``True``, create an underlying ``dist.ProcessGroup`` even for single-rank gangs.
        The gang is faked otherwise.
    :param dim_descriptions:
        String descriptions of returned gangs, used in log and error messages.

    :returns:
        A ``dict`` of two gangs; 0 maps to the gang of 2D mesh row,
        1 maps to the gang of the 2D mesh column.
    """
    row_count = root_gang.size // row_length

    mesh = torch.arange(root_gang.size).view(row_count, row_length)

    # Get the coordinate of this process in the mesh.
    rank_coords = [x.item() for x in torch.where(mesh == root_gang.rank)]
    mesh_shape = mesh.size()

    output = {}

    log.info(
        "Initializing sub-gangs for a 2D device mesh of shape {}.", list(mesh_shape)
    )
    if dim_descriptions is None:
        dim_descriptions = [f"dim-{dim}" for dim in range(2)]

    for dim in range(2):
        current_subgang: Gang | None = None

        gang_size = mesh_shape[1 - dim]

        log.info(
            "Initializing {} gang with a size of {}.", dim_descriptions[dim], gang_size
        )

        # Match row length (dim 0) or column length (dim 1)
        match gang_size:
            case 1:
                if create_single_rank_process_groups:
                    current_subgang = root_gang.make_gang([root_gang.rank])
                else:
                    current_subgang = FakeGang(device=root_gang.device)
            case root_gang.size:
                current_subgang = root_gang
            case _:
                # Create 1 gang per row (dim 0) or per column (dim 1)
                for i in range(mesh_shape[dim]):
                    ranks = mesh[i, :] if dim == 0 else mesh[:, i]
                    sub_gang = root_gang.make_gang(ranks.tolist())
                    if i == rank_coords[dim]:
                        current_subgang = sub_gang

        if current_subgang is None:
            raise InternalError(f"`current_gang` ({dim_descriptions[dim]}) is `None`.")

        output[dim] = current_subgang

    return output


def setup_hybrid_fsdp_gangs(gang: Gang, local_world_size: int) -> tuple[Gang, Gang]:
    """Make gangs to be used for hybrid-sharding FSDP.

    For instance; if we have 8 devices denoted by g0 to g7 and ``local_world_size``
    is 4, this function will make 2 sharding gangs and 4 replication gangs:

        2 sharding gangs of size 4:
            [g0, g1, g2, g3], [g4, g5, g6, g7]
        4 replication gangs of size 2:
            [g0, g4], [g1, g5], [g2, g6], [g3, g7]

    For efficiency, the caller should make sure adjacent ranks are on the same
    host.

    :param gang:
        The gang over which to shard and replicate.
    :param local_world_size:
        ``gang`` will be split into sub-gangs each containing
        ``local_world_size`` number of consecutive processes.
        The model will be fully sharded within each sub-gang and
        will be replicated across sub-gangs.

    :returns:
        A pair of two gangs: the sharding gang that the current process is
        part of, and the replication gang that the current process is part of
    """
    if local_world_size < 1:
        raise ValueError(
            f"`local_world_size` must be greater than 1, but is {local_world_size} instead."
        )

    if local_world_size == 1:
        raise GangError(
            f"`local_world_size` must be greater than 1, but is {local_world_size} instead. This hybrid configuration would force FSDP to switch to use `NO_SHARD`, which is deprecated. Please use DDP instead."
        )

    if local_world_size > gang.size:
        raise ValueError(
            f"`local_world_size` must be less than or equal to `gang.size` ({gang.size}), but is {local_world_size} instead."
        )

    if gang.size % local_world_size != 0:
        raise GangError(
            f"`gang.size` ({gang.size}) must be a multiple of `local_world_size` ({local_world_size})."
        )

    sub_gangs = _setup_2D_mesh_gangs(
        gang,
        row_length=local_world_size,
        create_single_rank_process_groups=True,
        dim_descriptions=["sharding", "replication"],
    )

    return sub_gangs[0], sub_gangs[1]


def setup_parallel_gangs(root_gang: Gang, *, tp_size: int = 1) -> Gangs:
    """Make gangs to be used for data and tensor parallelism.

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

    :param root_gang:
        The gang whose topology will be used to make the new gangs.
    :param tp_size:
        The size of tensor parallel gangs.

    :returns:
        Three gangs: the root gang, the data parallel gang that this
        process is part of, and the tensor parallel gang that this process is
        part of.
    """
    if tp_size <= 0:
        raise ValueError(f"`tp_size` must be greater than 0, but is {tp_size} instead.")

    if root_gang.size % tp_size != 0:
        raise GangError(
            f"The number of processes in the root gang is expected to be a multiple of the tensor parallel size ({tp_size}), but is {root_gang.size} instead."
        )

    output_from_2D_mesh = _setup_2D_mesh_gangs(
        root_gang,
        row_length=tp_size,
        dim_descriptions=["tensor parallel", "data parallel"],
    )

    return Gangs(root_gang, output_from_2D_mesh[1], output_from_2D_mesh[0])


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


def get_world_size() -> int:
    """Return the world size of the running job."""
    value = get_int_from_env("WORLD_SIZE")

    return 1 if value is None else value


def get_rank() -> int:
    """Return the rank of this process in the running job."""
    value = get_int_from_env("RANK", allow_zero=True)

    return 0 if value is None else value


def get_local_world_size() -> int:
    """Return the local world size of the running job."""
    value = get_int_from_env("LOCAL_WORLD_SIZE")

    return 1 if value is None else value


def get_local_rank() -> int:
    """Return the local rank of this process in the running job."""
    value = get_int_from_env("LOCAL_RANK", allow_zero=True)

    return 0 if value is None else value


def is_torchrun() -> bool:
    """Return ``True`` if this process was spawned by torchrun."""
    return "TORCHELASTIC_RUN_ID" in os.environ
