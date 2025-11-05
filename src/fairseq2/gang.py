# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
A :class:`Gang` represents a set of processes that can perform collective
operations such as all-reduce, broadcast, and other distributed primitives.

This module provides :class:`Gang` implementations that supports both real
distributed environments (using PyTorch's distributed backend) and simulated
environments for testing and single-process scenarios.

See :doc:`/concepts/gang` for more information.
"""

from __future__ import annotations

import os
import threading
import warnings
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Any, NoReturn, cast, final

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import Backend, ProcessGroup, ReduceOp
from typing_extensions import override

from fairseq2.device import Device
from fairseq2.error import (
    InternalError,
    InvalidOperationError,
    NotSupportedError,
    OperationalError,
)
from fairseq2.logging import log
from fairseq2.runtime.closable import Closable
from fairseq2.utils.tensor import to_tensor


class Gang(Closable):
    """Represents a set of processes that work collectively."""

    @abstractmethod
    def create_gang(self, ranks: Sequence[int]) -> Gang | None:
        """
        Creates a new sub-gang with the specified process ranks.

        The ranks must be unique and within the range [0, gang.size).

        Returns ``None`` if the current process is not included in ``ranks``.

        :raises ValueError: If ``ranks`` contains duplicates, or has one or more
            out of range values.

        :raises GangError: If the collective operation fails due to an unexpected
            error such as a network communication failure.

        .. code:: python
            :caption: Creating a sub-gang with specific processes

            # Create a gang with ranks 0, 2, 4 from an 8-process gang
            sub_gang = root_gang.create_gang([0, 2, 4])
            if sub_gang is not None:
                # Current process is part of the new gang
                print(f"New gang rank: {sub_gang.rank}, size: {sub_gang.size}")
        """

    @abstractmethod
    def as_process_group(self) -> ProcessGroup:
        """
        Returns this gang as a PyTorch ProcessGroup that can be used with
        PyTorch's distributed operations and collective communication functions.

        :raises NotSupportedError: If the gang implementation does not support
            conversion to a ProcessGroup (e.g. :class:`FakeGang`).
        """

    @abstractmethod
    def barrier(self) -> None:
        """
        Synchronizes all processes in the gang.

        This is a collective operation that blocks until all processes in the
        gang reach this synchronization point. Used for ensuring a consistent
        state across all processes before proceeding.

        :raises GangError: If the collective operation fails due to an unexpected
            error such as a network communication failure.
        """

    @abstractmethod
    def all_reduce(self, tensor: Tensor, op: ReduceOperation) -> None:
        """
        Reduces ``tensor`` across all processes using the specified operation.

        All-reduce combines tensors from all processes using the specified
        operation and distributes the result to all processes. The input tensor
        is modified in-place to contain the reduction result.

        :raises GangError: If the collective operation fails due to an unexpected
            error such as a network communication failure.

        .. code:: python
            :caption: Computing sum across all processes

            import torch
            from fairseq2.gang import ReduceOperation

            # Each process has a different tensor
            tensor = torch.tensor([gang.rank], dtype=torch.float32)

            # Sum across all processes
            gang.all_reduce(tensor, ReduceOperation.SUM)

            # Now tensor contains the sum of all ranks
            print(f"Sum of all ranks: {tensor.item()}")
        """

    @abstractmethod
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor) -> None:
        """
        Gathers tensors from all processes and puts them in ``output_tensor``.

        All-gather collects input tensors from all processes, concatenates them
        along a new first dimension in rank order and writes to the output
        tensor. The output tensor must have shape ``[gang.size, *input_tensor.shape]``
        and be contiguous in memory.

        :raises GangError: If the collective operation fails due to an unexpected
            error such as a network communication failure.

        .. code:: python
            :caption: Gathering tensors from all processes

            import torch

            # Each process contributes a tensor with its rank
            input_tensor = torch.tensor([gang.rank * 10], dtype=torch.float32)

            # Prepare output tensor for all gathered tensors
            output_tensor = torch.empty([gang.size, 1], dtype=torch.float32)

            # Gather from all processes
            gang.all_gather(output_tensor, input_tensor)

            # output_tensor now contains [0, 10, 20, ...] for ranks 0,1,2,...
        """

    @abstractmethod
    def all_gather_to_list(
        self, output_tensors: list[Tensor], input_tensor: Tensor
    ) -> None:
        """
        Gathers tensors from all processes and puts them in ``output_tensors``.

        Similar to :meth:`all_gather`, but stores the gathered tensors in a list
        instead of concatenating them into a single tensor. ``output_tensors``
        must be a pre-allocated list with length equal to ``gang.size``.

        :raises GangError: If the collective operation fails due to an unexpected
            error such as a network communication failure.
        """

    @abstractmethod
    def broadcast(self, tensor: Tensor, source_rank: int = 0) -> None:
        """
        Broadcasts ``tensor`` from the specified rank to all processes.

        Broadcast copies the tensor from the source process to all other
        processes. The tensor is modified in-place on non-source processes to
        contain the broadcasted data.

        ``source_rank`` must be in range [0, gang.size).

        :raises ValueError: If ``source_rank`` is out of valid range.

        :raises GangError: If the collective operation fails due to an unexpected
            error such as a network communication failure.
        """

    @abstractmethod
    def broadcast_objects(self, objects: list[object], source_rank: int = 0) -> None:
        """
        Broadcasts picklable ``objects`` from the specified rank to all processes.

        Similar to :meth:`broadcast`, but copies arbitrary Python objects that
        can be pickled. The objects are modified in-place on non-source processes.
        Each process must provide lists of equal sizes.

        ``source_rank`` must be in range [0, gang.size).

        :raises ValueError: If ``source_rank`` is out of valid range.

        :raises GangError: If the collective operation fails due to an unexpected
            error such as a network communication failure.
        """

    @property
    @abstractmethod
    def rank(self) -> int: ...

    @property
    @abstractmethod
    def size(self) -> int: ...

    @property
    @abstractmethod
    def device(self) -> Device: ...

    @property
    @abstractmethod
    def supports_process_group(self) -> bool: ...


class ReduceOperation(Enum):
    """
    Defines the standard reduction operations that can be performed across
    processes during collective communication operations like all-reduce.
    """

    SUM = 1
    MEAN = 2
    PRODUCT = 3
    MIN = 4
    MAX = 5


class GangError(Exception):
    """Raised when a collective communication error occurs."""


def raise_operational_gang_error(cause: GangError) -> NoReturn:
    """
    Raises an :class:`OperationalError` caused by a collective communication error.
    """
    raise OperationalError("A collective communication error occurred.") from cause


@final
class FakeGang(Gang):
    """
    Represents a non-distributed gang for local use.

    This implementation simulates gang operations without actual distributed
    communication, making it useful for testing, debugging, and single-process
    execution. All collective operations are no-ops.

    .. code:: python
        :caption: Simulating a collective operation

        import torch

        from fairseq2.gang import FakeGang

        device = torch.device("cpu")

        gang = FakeGang(device, rank=0, size=8)

        tensor = torch.tensor([gang.rank], dtype=torch.float32)

        # Simulates as if a real all-reduce operation is performed on the gang.
        gang.all_reduce(tensor, ReduceOperation.SUM)
    """

    def __init__(self, device: Device, *, rank: int = 0, size: int = 1) -> None:
        if size <= 0:
            raise ValueError("`size` must be greater than 0.")

        if rank < 0 or rank >= size:
            raise ValueError(
                f"`rank` must be greater than or equal to 0 and less than `size` ({size}), but is {rank} instead."
            )

        if device.type == "meta":
            raise ValueError("`device` must be a real device.")

        self._rank = rank
        self._size = size
        self._device = device

    @override
    def close(self) -> None:
        """No-op"""

    @override
    def create_gang(self, ranks: Sequence[int]) -> FakeGang | None:
        _check_ranks(ranks, self._size)

        try:
            idx = ranks.index(self._rank)
        except ValueError:
            return None

        return FakeGang(rank=idx, size=len(ranks), device=self._device)

    @override
    def as_process_group(self) -> ProcessGroup:
        raise NotSupportedError(
            f"`{FakeGang}` does not support conversion to a process group."
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
                    f"`op` must be a supported reduce operation, but is `{op.name}` instead."
                )

    @override
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor) -> None:
        if not output_tensor.is_contiguous():
            raise ValueError("`output_tensor` must be contiguous.")

        if output_tensor.ndim != input_tensor.ndim + 1:
            raise ValueError(
                "`output_tensor` must have a shape that is compatible with all-gather."
            )

        if output_tensor.size(0) != self._size:
            raise ValueError(
                f"Size of the first dimension of `output_tensor` must match the number of processes in the gang ({self._size}), but is {output_tensor.size(0)} instead."
            )

        for i in range(self._size):
            output_tensor[i].copy_(input_tensor)

    @override
    def all_gather_to_list(
        self, output_tensors: list[Tensor], input_tensor: Tensor
    ) -> None:
        if len(output_tensors) != self._size:
            raise ValueError(
                f"Length of `output_tensors` must match the number of processes in the gang ({self._size}), but is {len(output_tensors)} instead."
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

    @property
    @override
    def rank(self) -> int:
        return self._rank

    @property
    @override
    def size(self) -> int:
        return self._size

    @property
    @override
    def device(self) -> Device:
        return self._device

    @property
    @override
    def supports_process_group(self) -> bool:
        return False


@final
class ProcessGroupGang(Gang):
    """
    Represents a gang that wraps a PyTorch ProcessGroup.

    This is a distributed gang implementation that uses PyTorch's distributed
    backend for actual inter-process communication.
    """

    def __init__(self, _pg: ProcessGroup, _device: Device) -> None:
        """:meta private:"""
        self._pg = _pg
        self._device = _device

    @classmethod
    def create_default_process_group(
        cls,
        device: Device,
        *,
        timeout: timedelta | None = None,
        high_priority: bool = False,
    ) -> ProcessGroupGang:
        """
        Initializes the default process group and wraps it as a gang.

        For CUDA devices, NCCL; for CPU devices, Gloo backend will be used.

        ``timeout`` specifies the timeout for collective operations. If ``None``,
        the default timeout (15 minutes) will be used.

        If ``high_priority`` is ``True``, the underlying collective operations
        will be performed on high-priority channels (e.g. CUDA streams) if
        supported by the underlying backend.

        :raises ValueError: If ``device`` is not of type ``cpu`` or ``cuda``.

        :raises NotSupportedError: If ``torch.distributed`` is not available.

        :raises InvalidOperationError: If the root process group is already
            initialized.

        :raises GangError: If the underlying process group fails to initialize
            due to an unexpected error such as a network communication failure.
        """
        if log.is_enabled_for_debug():
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

            dist.set_debug_level_from_env()

        if not dist.is_available():
            raise NotSupportedError("torch.distributed is not available.")

        if dist.is_initialized():
            raise InvalidOperationError(
                "The root process group is already initialized."
            )

        backend: str | None

        if device.type == "cpu":
            backend = Backend.GLOO
        elif device.type == "cuda":
            backend = Backend.NCCL
        else:
            raise ValueError(
                f"`device` must be of type `cpu` or `cuda`, but is of type `{device.type}` instead."
            )

        if device.type == "cuda":
            # See https://github.com/pytorch/pytorch/issues/46874.
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

        if timeout is None:
            timeout = timedelta(minutes=15)

        kwargs: dict[str, Any] = {}

        pg_options = None

        if device.type == "cuda":
            # Eager process group initialization requires device index to be set.
            if device.index is None:
                device = Device("cuda", index=0)

            # Forces eager NCCL initialization.
            kwargs["device_id"] = device

            # If enabled, uses high-priority CUDA streams for NCCL.
            if high_priority:
                # Not available unless PyTorch is built with NCCL.
                from torch.distributed import ProcessGroupNCCL

                pg_options = ProcessGroupNCCL.Options(is_high_priority_stream=True)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore", message=".*options._timeout was specified.*"
                )

                dist.init_process_group(
                    backend, timeout=timeout, pg_options=pg_options, **kwargs
                )
        except (RuntimeError, ValueError) as ex:
            raise GangError("Underlying process group failed to initialize.") from ex

        pg = dist.group.WORLD
        if pg is None:
            raise OperationalError(
                "Root process group is not available after initialization."
            )

        return ProcessGroupGang(pg, device)

    @override
    def close(self) -> None:
        """Destroys the underlying ProcessGroup."""
        dist.destroy_process_group(self._pg)

    @override
    def create_gang(self, ranks: Sequence[int]) -> ProcessGroupGang | None:
        _check_ranks(ranks, self.size)

        if self._pg is not dist.group.WORLD:
            raise InvalidOperationError(
                "`create_gang()` must be called on the gang of the default process group."
            )

        try:
            backend = dist.get_backend()
        except RuntimeError as ex:
            raise OperationalError(
                "Root process group backend cannot be determined."
            ) from ex

        try:
            pg = dist.new_group(ranks, backend=backend)
        except RuntimeError as ex:
            s = ", ".join(sorted(str(r) for r in ranks))

            raise GangError(
                f"Child process group for ranks {s} cannot be created."
            ) from ex

        if self.rank not in ranks:
            return None

        return ProcessGroupGang(pg, self._device)

    @override
    def as_process_group(self) -> ProcessGroup:
        return self._pg

    @override
    def barrier(self) -> None:
        if self._device.type == "cuda":
            device_ids = [self._device.index]
        else:
            device_ids = None

        try:
            dist.barrier(group=self._pg, device_ids=device_ids)
        except RuntimeError as ex:
            self._raise_op_failed(ex, "barrier")

    @override
    def all_reduce(self, tensor: Tensor, op: ReduceOperation) -> None:
        try:
            dist.all_reduce(tensor, self._get_reduce_op(op), group=self._pg)
        except RuntimeError as ex:
            self._raise_op_failed(ex, "all_reduce")

    @override
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor) -> None:
        try:
            dist.all_gather_into_tensor(output_tensor, input_tensor, group=self._pg)
        except RuntimeError as ex:
            self._raise_op_failed(ex, "all_gather")

    @override
    def all_gather_to_list(
        self, output_tensors: list[Tensor], input_tensor: Tensor
    ) -> None:
        try:
            dist.all_gather(output_tensors, input_tensor, group=self._pg)
        except RuntimeError as ex:
            self._raise_op_failed(ex, "all_gather_to_list")

    @override
    def broadcast(self, tensor: Tensor, source_rank: int = 0) -> None:
        try:
            dist.broadcast(tensor, source_rank, group=self._pg)
        except RuntimeError as ex:
            self._raise_op_failed(ex, "broadcast")

    @override
    def broadcast_objects(self, objects: list[object], source_rank: int = 0) -> None:
        try:
            dist.broadcast_object_list(objects, source_rank, group=self._pg)
        except RuntimeError as ex:
            self._raise_op_failed(ex, "broadcast_object_list")

    @staticmethod
    def _raise_op_failed(cause: Exception, op: str) -> NoReturn:
        raise GangError(f"`{op}()` collective operation failed.") from cause

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
            f"`op` must be a supported reduce operation, but is `{op.name}` instead."
        )

    @property
    @override
    def rank(self) -> int:
        return dist.get_rank(self._pg)

    @property
    @override
    def size(self) -> int:
        return dist.get_world_size(self._pg)

    @property
    @override
    def device(self) -> Device:
        return self._device

    @property
    @override
    def supports_process_group(self) -> bool:
        return True


def _check_ranks(ranks: Sequence[int], gang_size: int) -> None:
    if len(set(ranks)) != len(ranks):
        raise ValueError("Ranks in `ranks` must be unique.")

    for idx, rank in enumerate(ranks):
        if rank < 0 or rank > gang_size:
            raise ValueError(
                f"Rank at index {idx} in `ranks` must be greater than or equal to 0 and less than the size of the gang ({gang_size}), but is {rank} instead."
            )


@dataclass(kw_only=True, frozen=True)
class Gangs(Closable):
    """
    Holds parallel gangs used in distributed configurations.

    Each gang is used for a different parallelism strategy such as data, tensor,
    or pipeline parallelism.

    Check out :func:`create_parallel_gangs` and :func:`create_fsdp_gangs` to see
    how to initialize a ``Gangs`` instance.
    """

    root: Gang
    """The root gang containing all processes."""

    dp: Gang
    """The data parallel gang."""

    rdp: Gang
    """
    The replicated data parallel gang (i.e. inter-node for HSDP).

    This is a sub-gang of :attr:`dp` used for replicated data parallelism. In
    PyTorch, this gang is used by DDP as well as by FSDP for inter-node
    communication when hybrid sharding is enabled.
    """

    sdp: Gang
    """
    The sharded data parallel gang (i.e. intra-node for HSDP).

    This is a sub-gang of :attr:`dp` used for sharded data parallelism. In
    PyTorch, this gang is used by FSDP. If hybrid sharding is enabled, it will
    be used only for intra-node communication, while inter-node communication
    will be handled by :attr:`rdp`.
    """

    tp: Gang
    """The tensor parallel gang."""

    pp: Gang
    """The pipeline parallel gang."""

    def __post_init__(self) -> None:
        if self.root.rank == 0:
            if self.dp.rank != 0 or self.tp.rank != 0 and self.pp.rank != 0:
                raise ValueError(
                    "Coordinator process of the root gang (i.e. `root.rank == 0`) must be rank 0 in all parallel gangs."
                )

    def __enter__(self) -> None:
        _thread_local.current_gangs.append(self)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _thread_local.current_gangs.pop()

    def close(self) -> None:
        """Destroys all gangs."""
        self.root.close()


_thread_local = threading.local()

# Holds the stack of current thread-local gangs.
_thread_local.current_gangs = []


def maybe_get_current_gangs() -> Gangs | None:
    """
    Returns the current gangs to use for collective operations.

    By default, this function returns ``None``. The current gangs of the calling
    thread can be set by using :class:`Gangs` as a context manager:

    .. code::

        from fairseq2.gang import Gangs

        gangs = Gangs(...)

        with gangs:
            current_gangs = maybe_get_current_gangs()

            assert current_gangs is gangs

        current_gangs = maybe_get_current_gangs()

        assert current_gangs is None

    Within fairseq2, this function is used by model factories to retrieve the
    current gangs and shard the constructed models accordingly. The current gangs
    are set internally by fairseq2 before calling the factories.

    Note that the return value of this function is thread specific. Individual
    threads may have their own set of current gangs.
    """
    if _thread_local.current_gangs:
        return cast(Gangs, _thread_local.current_gangs[-1])

    return None


def create_parallel_gangs(root_gang: Gang, *, tp_size: int = 1) -> Gangs:
    """
    Creates gangs to be used for data and model parallelism.

    For instance, if there are 8 devices denoted by d0 to d7 and 2 devices are
    used for tensor parallelism (i.e. ``tp_size`` is 2), this function will
    create 4 tensor parallel gangs and 2 data parallel gangs by splitting
    ``root_gang`` as:

    4 tensor parallel gangs:
        [d0, d1], [d2, d3], [d4, d5], [d6, d7]
    2 data parallel gangs:
        [d0, d2, d4, d6], [d1, d3, d5, d7]

    For efficiency, the caller should make sure adjacent ranks are on the same
    host. For example, if there are two hosts with a total of 16 GPUs, ranks 0
    to 7 belong to the first host and ranks 8 to 15 belong to the second host.

    .. note::

        If ``root_gang`` is a PyTorch ``ProcessGroup`` with NCCL backend, this
        function uses the experimental ``split_group`` API in PyTorch 2.5 and
        later. See `here`__ for more information.

    .. __: https://dev-discuss.pytorch.org/t/rfc-c10d-a-new-pytorch-api-split-group-to-create-a-process-group-through-ncclcommsplit/2233
    """
    if tp_size < 1:
        raise ValueError(
            f"`tp_size` must be greater than or equal to 1, but is {tp_size} instead."
        )

    if root_gang.size % tp_size != 0:
        raise ValueError(
            f"`tp_size` must be a factor of `root_gang.size` ({root_gang.size}), but is {tp_size} instead."
        )

    fake_gang = FakeGang(device=root_gang.device)

    dp_size = root_gang.size // tp_size

    mesh = torch.arange(root_gang.size).view(dp_size, tp_size)

    # Get the coordinate of this process in the mesh.
    rank_coord = [x.item() for x in torch.where(mesh == root_gang.rank)]

    dp_gang: Gang | None = None

    log.info("Creating data parallel gang with {} process(es).", dp_size)

    # Build the gangs for data parallelism.
    match dp_size:
        case 1:
            dp_gang = fake_gang
        case root_gang.size:
            dp_gang = root_gang
        case _:
            if root_gang.device.type == "cuda" and root_gang.supports_process_group:
                root_pg = root_gang.as_process_group()

                try:
                    pg = torch.distributed.distributed_c10d.split_group(
                        parent_pg=root_pg, split_ranks=mesh.T.tolist(), group_desc="dp"
                    )
                except RuntimeError as ex:
                    raise GangError(
                        "Data parallel process group cannot be created."
                    ) from ex

                if pg is not None:
                    dp_gang = ProcessGroupGang(pg, root_gang.device)
            else:
                for i in range(tp_size):
                    sub_gang = root_gang.create_gang(mesh[:, i].tolist())
                    if i == rank_coord[1]:
                        dp_gang = sub_gang

    if dp_gang is None:
        raise InternalError("`dp_gang` is `None`.")

    log.info("Data parallel gang created.")

    tp_gang: Gang | None = None

    log.info("Creating tensor parallel gang with {} process(es).", tp_size)

    # Build the gangs for tensor parallelism.
    match tp_size:
        case 1:
            tp_gang = fake_gang
        case root_gang.size:
            tp_gang = root_gang
        case _:
            if root_gang.device.type == "cuda" and root_gang.supports_process_group:
                root_pg = root_gang.as_process_group()

                try:
                    pg = torch.distributed.distributed_c10d.split_group(
                        parent_pg=root_pg, split_ranks=mesh.tolist(), group_desc="tp"
                    )
                except RuntimeError as ex:
                    raise GangError(
                        "Tensor parallel process group cannot be created."
                    ) from ex

                if pg is not None:
                    tp_gang = ProcessGroupGang(pg, root_gang.device)
            else:
                for i in range(dp_size):
                    sub_gang = root_gang.create_gang(mesh[i, :].tolist())
                    if i == rank_coord[0]:
                        tp_gang = sub_gang

    if tp_gang is None:
        raise InternalError("`tp_gang` is `None`.")

    log.info("Tensor parallel gang created.")

    # TODO: implement!
    log.info("Creating pipeline parallel gang with {} process(es).", 1)

    log.info("Pipeline parallel gang created.")

    return Gangs(
        root=root_gang, dp=dp_gang, rdp=dp_gang, sdp=fake_gang, tp=tp_gang, pp=fake_gang
    )


def create_fsdp_gangs(gangs: Gangs, intra_node_size: int | None = None) -> Gangs:
    """
    Creates gangs to be used for hybrid or fully sharded data parallelism.

    If ``intra_node_size`` is ``None``, :attr:`Gangs.sdp` (sharded data gang)
    will be set to the same gang as :attr:`Gangs.dp` and :attr:`Gangs.rdp`
    (replicated data gang) will be set to a fake gang of size 1. This topology
    represents a fully sharded data parallel strategy.

    An integer ``intra_node_size`` indicates hybrid sharded data parallelism.
    For instance, if there are 8 devices denoted by d0 to d7 and 4 devices are
    used for intra-node parallelism (i.e. ``intra_node_size`` is 4), this
    function will create 2 intra-node gangs and 4 inter-node gangs by splitting
    ``gangs.dp`` as:

    2 intra-node gangs of size 4:
        [d0, d1, d2, d3], [d4, d5, d6, d7]
    4 inter-node gangs of size 2:
        [d0, d4], [d1, d5], [d2, d6], [d3, d7]

    For efficiency, the caller should make sure adjacent ranks are on the same
    host.

    At the end of the call, ``gangs.rdp`` (replicated data gang) will point to
    the inter-node gang and ``gangs.sdp`` (sharded data gang) will point to the
    intra-node gang.

    Returns the same :class:`Gangs` instance passed to ``gangs`` with its
    :attr:`Gangs.rdp` and :attr:`Gangs.sdp` attributes set accordingly.

    .. note::

        If ``root_gang`` is a PyTorch ``ProcessGroup`` with NCCL backend, this
        function uses the experimental ``split_group`` API in PyTorch 2.5 and
        later. See `here`__ for more information.

    .. __: https://dev-discuss.pytorch.org/t/rfc-c10d-a-new-pytorch-api-split-group-to-create-a-process-group-through-ncclcommsplit/2233
    """
    if intra_node_size is None:
        fake_gang = FakeGang(gangs.root.device)

        return Gangs(
            root=gangs.root,
            dp=gangs.dp,
            rdp=fake_gang,
            sdp=gangs.dp,
            tp=gangs.tp,
            pp=gangs.pp,
        )

    if intra_node_size <= 1:
        raise ValueError(
            f"`intra_node_size` must be greater than 1, but is {intra_node_size} instead."
        )

    dp_gang = gangs.dp

    if dp_gang.size % intra_node_size != 0:
        raise ValueError(
            f"`intra_node_size` must be a factor of `gangs.dp.size` ({dp_gang.size}), but is {intra_node_size} instead."
        )

    fake_gang = FakeGang(device=dp_gang.device)

    inter_node_size = dp_gang.size // intra_node_size

    mesh = torch.arange(dp_gang.size).view(inter_node_size, intra_node_size)

    # Get the coordinate of this process in the mesh.
    rank_coord = [x.item() for x in torch.where(mesh == dp_gang.rank)]

    inter_gang: Gang | None = None

    log.info("Creating inter-node data parallel gang with {} process(es).", inter_node_size)  # fmt: skip

    # Build the gangs for inter-node data parallelism.
    match inter_node_size:
        case 1:
            inter_gang = fake_gang
        case dp_gang.size:
            inter_gang = dp_gang
        case _:
            if dp_gang.device.type == "cuda" and dp_gang.supports_process_group:
                dp_pg = dp_gang.as_process_group()

                try:
                    pg = torch.distributed.distributed_c10d.split_group(
                        parent_pg=dp_pg, split_ranks=mesh.T.tolist(), group_desc="inter"
                    )
                except RuntimeError as ex:
                    raise GangError(
                        "Inter-node process group cannot be created."
                    ) from ex

                if pg is not None:
                    inter_gang = ProcessGroupGang(pg, dp_gang.device)
            else:
                for i in range(intra_node_size):
                    sub_gang = dp_gang.create_gang(mesh[:, i].tolist())
                    if i == rank_coord[1]:
                        inter_gang = sub_gang

    if inter_gang is None:
        raise InternalError("`inter_gang` is `None`.")

    intra_gang: Gang | None = None

    log.info("Creating intra-node data parallel gang with {} process(es).", intra_node_size)  # fmt: skip

    # Build the gangs for intra-node data parallelism.
    match intra_node_size:
        case 1:
            intra_gang = fake_gang
        case dp_gang.size:
            intra_gang = dp_gang
        case _:
            if dp_gang.device.type == "cuda" and dp_gang.supports_process_group:
                dp_pg = dp_gang.as_process_group()

                try:
                    pg = torch.distributed.distributed_c10d.split_group(
                        parent_pg=dp_pg, split_ranks=mesh.tolist(), group_desc="intra"
                    )
                except RuntimeError as ex:
                    raise GangError(
                        "Intra-node process group cannot be created."
                    ) from ex

                if pg is not None:
                    intra_gang = ProcessGroupGang(pg, dp_gang.device)
            else:
                for i in range(inter_node_size):
                    sub_gang = dp_gang.create_gang(mesh[i, :].tolist())
                    if i == rank_coord[0]:
                        intra_gang = sub_gang

    if intra_gang is None:
        raise InternalError("`intra_gang` is `None`.")

    return Gangs(
        root=gangs.root,
        dp=dp_gang,
        rdp=inter_gang,
        sdp=intra_gang,
        tp=gangs.tp,
        pp=gangs.pp,
    )


def create_fake_gangs(device: Device) -> Gangs:
    """
    Creates a set of fake gangs for single-process scenarios.

    This is a helper function where every :class:`FakeGang` is initialized with
    rank 0 and world size 1. For more complex simulated/testing environments,
    :class:`FakeGang` instances can be individually constructed per parallelism
    strategy and passed to a :class:`Gangs` object.

    .. code:: python
        :caption: Creating fake gangs for testing

        import torch

        from fairseq2.gang import create_fake_gangs

        device = torch.device("cpu")

        gangs = create_fake_gangs(device)

        tensor = torch.tensor([gang.rank], dtype=torch.float32)

        # Simulates as if a real all-reduce operation is performed on the data
        # parallel gang.
        gangs.dp.all_reduce(tensor, ReduceOperation.SUM)
    """
    gang = FakeGang(device=device)

    return Gangs(root=gang, dp=gang, rdp=gang, sdp=gang, tp=gang, pp=gang)


def broadcast_flag(gang: Gang, flag: bool, source_rank: int = 0) -> bool:
    """
    Broadcasts a boolean flag to all processes in ``gang`` from the specified
    rank.

    Returns the broadcasted boolean value on all processes.

    .. code:: python
        :caption: Broadcasting a flag across processes

        # Only rank 0 sets the flag
        should_continue = gang.rank == 0 and some_condition()

        # Broadcast the decision to all processes
        should_continue = broadcast_flag(gang, should_continue, source_rank=0)

        if should_continue:
            # All processes execute this together
            continue_processing()
    """
    flag_pt = to_tensor(flag, device=gang.device)

    gang.broadcast(flag_pt, source_rank)

    return bool(flag_pt)


def all_sum(gang: Gang, value: float | int) -> Tensor:
    """
    Sums a scalar value over all processes in ``gang``.

    Returns a tensor containing the sum of all process values.

    .. code:: python
        :caption: Computing total loss across processes

        # Each process computes its local loss
        local_loss = compute_loss(batch)

        # Sum losses across all processes
        total_loss = all_sum(gang, local_loss)

        # Now `total_loss` contains the sum from all processes
        average_loss = total_loss / gang.size
    """
    value_pt = to_tensor(value, device=gang.device)

    gang.all_reduce(value_pt, ReduceOperation.SUM)

    return value_pt
