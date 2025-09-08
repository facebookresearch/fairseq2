=============
fairseq2.gang
=============

.. currentmodule:: fairseq2.gang

What's Gang?
^^^^^^^^^^^^

A Gang represents a group of processes (`e.g.`, GPUs) that work together in a distributed setting.
Each Gang:

- Has a unique rank for each process
- Knows its total size (number of processes)
- Supports collective operations (`e.g.`, ``all_reduce``, ``broadcast``)
- Is associated with a specific device (CPU or CUDA)

The gang module provides distributed computing primitives for fairseq2, enabling
efficient coordination and communication between multiple processes in distributed
training scenarios.

Core Classes
============

.. autoclass:: Gang
   :members:
   :undoc-members:
   :show-inheritance:

Gang represents a set of processes that work collectively. It provides an abstract
interface for collective communication operations like all-reduce, broadcast, and
all-gather.

Gang Implementations
====================

.. autoclass:: FakeGang
   :members:
   :undoc-members:
   :show-inheritance:

FakeGang is used for local, non-distributed scenarios. It simulates gang behavior
for single-process execution.

.. autoclass:: ProcessGroupGang
   :members:
   :undoc-members:
   :show-inheritance:

ProcessGroupGang wraps PyTorch's ProcessGroup to provide gang functionality in
distributed environments.

Gang Configuration
==================

.. autoclass:: Gangs
   :members:
   :undoc-members:
   :show-inheritance:

Gangs is a dataclass that holds different types of gangs used in parallel training:

- **root**: The root gang containing all processes
- **dp**: Data parallel gang
- **rdp**: Replicated data parallel gang (inter-node)
- **sdp**: Sharded data parallel gang (intra-node)
- **tp**: Tensor parallel gang
- **pp**: Pipeline parallel gang

Enums and Types
===============

.. autoclass:: ReduceOperation
   :members:
   :undoc-members:
   :show-inheritance:

ReduceOperation specifies the type of reduction to perform in all-reduce operations.

Factory Functions
=================

.. autofunction:: create_fake_gangs

Creates fake gangs for local, non-distributed execution.

.. autofunction:: create_parallel_gangs

Sets up gangs for data and tensor parallelism in distributed training.

.. autofunction:: create_fsdp_gangs

Sets up gangs for Fully Sharded Data Parallel (FSDP) training.

Utility Functions
=================

.. autofunction:: broadcast_flag

Broadcasts a boolean flag to all processes in a gang.

.. autofunction:: all_sum

Sums a value over all processes in a gang.

Examples
========

**Basic Gang Setup**

.. code-block:: python

    from fairseq2.gang import create_fake_gangs
    from fairseq2.device import Device

    # Create fake gangs for local development
    device = Device("cpu")
    gangs = create_fake_gangs(device)

    print(f"Root gang size: {gangs.root.size}")
    print(f"Data parallel gang size: {gangs.dp.size}")

**Distributed Training Setup**

.. code-block:: python

    from fairseq2.gang import ProcessGroupGang, create_parallel_gangs
    from fairseq2.device import get_default_device

    # Initialize distributed process group
    device = get_default_device()
    root_gang = ProcessGroupGang.create_default_process_group(device)

    # Create parallel gangs with tensor parallelism
    gangs = create_parallel_gangs(root_gang, tp_size=2)

    print(f"Process rank: {gangs.root.rank}")
    print(f"World size: {gangs.root.size}")
    print(f"TP gang size: {gangs.tp.size}")

Gang Topology
==============

fairseq2's gang system supports complex parallel training topologies:

**Data Parallelism**
   Multiple processes train on different data shards but maintain identical model copies.

**Tensor Parallelism**
   The model is split across multiple devices, with each device handling part of each layer.

**Pipeline Parallelism**
   Different layers of the model run on different devices in a pipeline fashion.

**Fully Sharded Data Parallelism (FSDP)**
   Combines data parallelism with parameter sharding, reducing memory usage while
   maintaining training efficiency.

The gang system automatically handles the communication patterns required for each
parallelism strategy, making it easy to scale training across many GPUs and nodes.

See Also
========

* :doc:`device` - Device management utilities
