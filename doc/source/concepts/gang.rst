===============
What is a Gang?
===============

.. currentmodule:: fairseq2.gang

A :class:`Gang` represents a set of processes that can perform collective
operations such as all-reduce, broadcast, and other distributed primitives.
:mod:`fairseq2.gang` module provides gang implementations that supports both
real distributed environments (using PyTorch's distributed backend - see
:class:`ProcessGroupGang`) and simulated environments for testing and
single-process scenarios (see :class:`FakeGang`).

Each gang operates on a specific device and maintains information about the
process ranks and the total number of processes.

.. code:: python
    :caption: Tensor Parallel Distributed Gang Setup

    import torch

    from fairseq2.gang import ProcessGroupGang, create_parallel_gangs
    from fairseq2.device import get_default_device

    # Get the default CUDA or CPU device of the process.
    device = get_default_device()

    # Depending on the device creates a ProcessGroup with NCCL or Gloo backend.
    root_gang = ProcessGroupGang.create_default_process_group(device)

    # Create data and tensor parallel gangs.
    gangs = create_parallel_gangs(root_gang, tp_size=8)

    print(f"Root Gang: {gangs.root.size}/{gangs.root.rank}")
    print(f"Tensor Parallel Gang: {gangs.tp.rank}/{gangs.tp.size}")

    tensor = torch.ones((8, 8), device=device)

    # All reduce on tensor parallel ranks.
    gangs.tp.all_reduce(tensor, ReduceOperation.SUM)

How is Gang different from PyTorch ProcessGroup?
================================================

:class:`Gang` provides an abstraction over ProcessGroup that consolidates all
collective communication operations into a single interface. This approach
offers several advantages:

- APIs can explicitly declare and encapsulate their distributed communication
  requirements within a cohesive gang object, rather than relying on implicit
  global state and scattered ``torch.distributed`` function calls.
- Device management, rank information, and collective operations are combined
  into a single cohesive object, eliminating the need to manually track process
  groups, devices, and ranks separately.
- Seamless testing and debugging through :class:`FakeGang`, which simulates
  distributed behavior in single-process environments without requiring actual
  multi-process setup or mocking distributed calls.
- A clean abstraction layer that allows swapping the underlying ProcessGroup
  implementation with high-performing, fault-tolerant alternatives without
  updating code that depends on :class:`Gang`.

.. code:: python
    :caption: Gang vs ProcessGroup usage comparison

    import torch

    # ProcessGroup approach - scattered state management
    import torch.distributed as dist

    pg = dist.new_group([0, 1, 2, 3])
    device = torch.device("cuda:0")
    rank = dist.get_rank(pg)

    # Multiple calls with separate state tracking
    dist.all_reduce(tensor, group=pg)
    dist.broadcast(tensor, src=0, group=pg)

    # Gang approach - unified interface
    from fairseq2.gang import ProcessGroupGang, create_parallel_gangs

    gang = ProcessGroupGang.create_default_process_group(device)

    # Everything encapsulated in one object
    gang.all_reduce(tensor, ReduceOperation.SUM)
    gang.broadcast(tensor, source_rank=0)

    # gang.rank, gang.size, gang.device all available

How is Gang different from PyTorch DeviceMesh?
==============================================

:class:`Gang` and DeviceMesh address the management of multi-dimensional
distributed computing with different design philosophies. DeviceMesh provides a
general-purpose multi-dimensional device topology abstraction where users work
with coordinates and dimensions. In contrast, :class:`Gang` uses the :class:`Gangs`
container to explicitly associate each gang with a specific parallelism strategy
(data, model, pipeline), providing a more structured approach to parallelism
management.

- :class:`Gang` uses explicit parallelism strategy associations through the
  :class:`Gangs` container (:attr:`Gangs.dp`, :attr:`Gangs.tp`, :attr:`Gangs.pp`),
  while DeviceMesh requires users to manually track which dimensions correspond
  to which parallelism strategies and calculate coordinates for specific
  operations.
- :class:`Gang` provides direct collective operation methods (:meth:`~Gang.all_reduce`,
  :meth:`~Gang.broadcast`, etc.) as first-class APIs on each :class:`Gang`
  instance, whereas DeviceMesh typically works with PyTorch's distributed tensor
  (DTensor) framework and requires additional abstractions for operations.
- :class:`Gang` includes built-in testing support through :class:`FakeGang` that
  can simulate all parallelism strategies uniformly, while DeviceMesh testing
  requires complex setup or mocking of multi-dimensional distributed environments.
- :class:`Gang` creates purpose-built process groups for each parallelism
  strategy through functions like :func:`create_parallel_gangs()` and
  :func:`create_fsdp_gangs()`, while DeviceMesh provides general mesh slicing
  operations that require additional logic to map to specific parallelism
  patterns.

.. code:: python
    :caption: Gang vs DeviceMesh usage comparison

    # DeviceMesh approach - coordinate-based access
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cuda", (2, 4))  # 2D mesh: 2x4

    # Manual coordinate tracking for parallelism strategies
    dp_mesh = mesh[:, 0]      # First column for data parallel
    tp_mesh = mesh[0, :]      # First row for tensor parallel

    # Gang approach - explicit parallelism strategies 
    from fairseq2.gang import ProcessGroupGang, create_parallel_gangs

    root_gang = ProcessGroupGang.create_default_process_group(device)

    gangs = create_parallel_gangs(root_gang, tp_size=4)

    tensor = torch.ones((8, 8), device=device)

    # Direct semantic access to parallelism-specific gangs
    gangs.dp.all_reduce(tensor, ReduceOperation.SUM)  # Data parallel
    gangs.tp.all_reduce(tensor, ReduceOperation.SUM)  # Tensor parallel

How to create a Gang?
=====================

For single-process jobs and testing environments, a quick and easy solution is
to create a :class:`FakeGang`, which simulates the entire API surface of the
:class:`Gang` interface. This approach is particularly beneficial when the same
code needs to run both standalone and in a distributed environment, as it does
not require any implementation changes.

.. code:: python
    :caption: Running in a simulated environment

    import torch

    from fairseq2.gang import FakeGang, ProcessGroupGang, ReduceOperation
    from fairseq2.device import get_default_device
    from fairseq2.utils.env import get_world_size

    def main() -> None:
        # Get the default CUDA or CPU device of the process.
        device = get_default_device()

        # Infer the world size from the `WORLD_SIZE` environment variable.
        world_size = get_world_size()

        # If only a single rank, create a fake gang.
        if world_size == 1:
            gang = FakeGang(device)
        else:
            gang = ProcessGroupGang.create_default_process_group(device)

        some_distributed_function(gang)


    def some_distributed_function(gang: Gang) -> None:
        tensor = torch.ones((8, 8), device=gang.device)

        # All reduce on ranks.
        gang.all_reduce(tensor, ReduceOperation.SUM)


For real distributed environments, fairseq2 provides :class:`ProcessGroupGang`,
which is a wrapper on top of PyTorch’s ProcessGroup. It encapsulates scattered
``torch.distributed`` collective calls within a cohesive and consolidated
:class:`Gang` interface.

.. code:: python
    :caption: Initialize PyTorch ProcessGroup

    import torch

    from fairseq2.gang import ProcessGroupGang
    from fairseq2.device import get_default_device

    # Get the default CUDA or CPU device of the process.
    device = get_default_device()

    # Depending on the device creates a ProcessGroup with NCCL or Gloo backend.
    gang = ProcessGroupGang.create_default_process_group(device)

    tensor = torch.ones((8, 8), device=gang.device)

    # All reduce on ranks.
    gang.all_reduce(tensor, ReduceOperation.SUM)

How to create Gangs for data and model parallelism?
===================================================

Most model architectures either require or greatly benefit from parallelism
strategies such as data parallelism or tensor parallelism during training and/or
inference. In fairseq2, each parallelism strategy is backed by an individual
:class:`Gang` instance. Because correctly instantiating parallel gangs can be a
non-trivial task, fairseq2 provides two helper functions: :func:`create_parallel_gangs`
and :func:`create_fsdp_gangs`.

The :func:`create_parallel_gangs` function takes a "root" gang as input and
splits it into sub-gangs, with each sub-gang representing a distinct parallelism
strategy. The topology of these gangs is determined by the specified size
arguments. The function returns a :class:`Gangs` instance, which serves as a
container for all the created parallel gangs.

.. code:: python
    :caption: Initialize data and model parallel gangs

    import torch

    from fairseq2.gang import ProcessGroupGang, create_parallel_gangs
    from fairseq2.device import get_default_device

    # Get the default CUDA or CPU device of the process.
    device = get_default_device()

    # Depending on the device creates a ProcessGroup with NCCL or Gloo backend.
    root_gang = ProcessGroupGang.create_default_process_group(device)

    # If there are 8 devices denoted by d0 to d7 and 2 devices are used for
    # tensor parallelism (i.e. `tp_size` is 2), this function will create 4
    # tensor parallel gangs and 2 data parallel gangs by splitting `root_gang`
    # as:
    #   4 tensor parallel gangs:
    #       [d0, d1], [d2, d3], [d4, d5], [d6, d7]
    #   2 data parallel gangs:
    #       [d0, d2, d4, d6], [d1, d3, d5, d7]
    gangs = create_parallel_gangs(root_gang, tp_size=8)

    tensor = torch.ones((8, 8), device=device)

    # All reduce on tensor parallel ranks.
    gangs.tp.all_reduce(tensor, ReduceOperation.SUM)

During training, a data parallel gang can further be split into replicated and
sharded gangs to support hybrid and fully sharded data parallelism strategies.

.. code:: python
    :caption: Initialize fully sharded data parallelism

    import torch

    from fairseq2.gang import ProcessGroupGang, create_parallel_gangs
    from fairseq2.device import get_default_device

    # Get the default CUDA or CPU device of the process.
    device = get_default_device()

    # Depending on the device creates a ProcessGroup with NCCL or Gloo backend.
    root_gang = ProcessGroupGang.create_default_process_group(device)

    # All size parameters (e.g. `tp_size`) for model parallelism strategies
    # default to 1. If there are 8 devices, this call will only instantiate a
    # new data parallel gang.
    gangs = create_parallel_gangs(root_gang)

    # At the end of the call, `gangs.sdp` (sharded) will point to the same gang
    # as `gangs.dp` and `gangs.rdp` (replicated) will be set to a fake gang of
    # size 1.
    gangs = create_fsdp_gangs(gangs)


.. code:: python
    :caption: Initialize hybrid sharded data parallelism

    import torch

    from fairseq2.gang import ProcessGroupGang, create_parallel_gangs
    from fairseq2.device import get_default_device

    # Get the default CUDA or CPU device of the process.
    device = get_default_device()

    # Depending on the device creates a ProcessGroup with NCCL or Gloo backend.
    root_gang = ProcessGroupGang.create_default_process_group(device)

    # All size parameters (e.g. `tp_size`) for model parallelism strategies
    # default to 1. If there are 8 devices, this call will only instantiate a
    # new data parallel gang.
    gangs = create_parallel_gangs(root_gang)

    # If 4 devices are used for intra-node parallelism, this function will
    # create 2 intra-node gangs and 4 inter-node gangs by splitting `gangs.dp`
    # as:
    #   2 intra-node gangs of size 4:
    #       [d0, d1, d2, d3], [d4, d5, d6, d7]
    #   4 inter-node gangs of size 2:
    #       [d0, d4], [d1, d5], [d2, d6], [d3, d7]
    #
    # At the end of the call, `gangs.rdp` (replicated) will point to the
    # inter-node gang and `gangs.sdp` will point to the intra-node gang.
    gangs = create_fsdp_gangs(gangs, intra_node_size=4)

How is Gang used in fairseq2?
=============================

fairseq2 is designed to allow researchers to begin their work on a single device
with simple experimentation code, which can later be gradually scaled up to
thousands of GPUs with minimal code changes. Consequently, several major APIs
such as :doc:`model loading </guides/add_model>`, :doc:`checkpointing </reference/fairseq2.model_checkpoint>`,
and :doc:`dataset reading </reference/datasets/index>` natively support scaling
and parallelism through the gang abstraction. Refer to the relevant guides to
learn more about how gangs are utilized there.

How to use Gangs in deeply nested functions?
============================================

fairseq2 provides a basic API for setting a :class:`Gangs` instance as the
"current" gangs for the calling thread. This feature is particularly useful in
procedural programming, as it eliminates the need to pass a :class:`Gangs`
instance through every function call.

When a :class:`Gangs` instance is used as a context manager, it is set as the
current gangs. You can nest ``with gangs`` statements to override the current
gangs as needed. The current gangs instance can be retrieved by calling the
:func:`maybe_get_current_gangs` function.

.. code:: python
    :caption: Set and retrieve current thread-local gangs

    from fairseq2.gang import Gangs, maybe_get_current_gangs

    gangs = Gangs(...)

    with gangs:
        current_gangs = maybe_get_current_gangs()

        assert current_gangs is gangs

    current_gangs = maybe_get_current_gangs()

    assert current_gangs is None
