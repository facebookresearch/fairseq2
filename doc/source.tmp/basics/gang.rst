.. _basics-gang:

:octicon:`table` Gang
=====================


Overview
--------

Gang is fairseq2's abstraction for distributed training that provides a clean interface for collective operations (`e.g.`, ``all_reduce``, ``all_gather``, and ``broadcast``) across processes in a distributed environment.
It simplifies PyTorch's distributed training while supporting both data parallelism and tensor parallelism.

This design encapsulates the complexity of PyTorch's ``torch.distributed`` while supporting:

- **Data Parallelism**: Distributing batches of data across multiple GPUs.
- **Tensor Parallelism**: Partitioning model tensors for efficient computation.
- **Flexible Process Grouping**: Organizing processes into groups dynamically.

Core Concepts
-------------

.. note::

    It would be helpful to understand the following concepts before diving into Gang:

    - `PyTorch Distributed <https://pytorch.org/tutorials/beginner/dist_overview.html>`_
    - `FSDP <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`_
    - `Distributed Device Mesh <https://pytorch.org/tutorials/recipes/distributed_device_mesh.html>`_

What's Gang?
^^^^^^^^^^^^

A Gang represents a group of processes (`e.g.`, GPUs) that work together in a distributed setting.
Each Gang:

- Has a unique rank for each process
- Knows its total size (number of processes)
- Supports collective operations (`e.g.`, ``all_reduce``, ``broadcast``)
- Is associated with a specific device (CPU or CUDA)

By abstracting the concept of "process groups" from PyTorch Distributed, Gangs make distributed training simpler and more expressive.

Types of Gangs
^^^^^^^^^^^^^^

1. **FakeGang**

   - A non-distributed gang for single-process execution
   - Useful for local development and debugging
   - Emulates distributed operations locally

2. **ProcessGroupGang**

   - Wraps PyTorch's ProcessGroup for actual distributed training
   - Supports both NCCL (for GPU) and Gloo (for CPU) backends
   - Handles monitored barriers and collective operations (e.g., `all_reduce`, `all_gather`, `broadcast`)
   - Supports configurable timeouts for detecting deadlocks when using monitored barriers

Distributed Training Basics
---------------------------

Key Terms
^^^^^^^^^

1. **World Size**: The total number of processes participating in distributed training.
2. **Rank**: The unique ID of a process within the world.
3. **Device**: The hardware (CPU/GPU) associated with each process
4. **Process Group**: A subset of processes for performing collective operations.

Collective Operations
^^^^^^^^^^^^^^^^^^^^^

The Gang interface supports the following methods:

.. code-block:: python

    # Reduce tensor across processes
    gang.all_reduce(tensor, ReduceOperation.SUM)

    # Gather tensors from all processes
    gang.all_gather(output_tensor, input_tensor)

    # Gather tensors from all processes into a list
    gang.all_gather_to_list(output_tensors, input_tensor)

    # Broadcast tensor from source rank to all others
    gang.broadcast(tensor, source_rank=0)

    # Synchronize all processes
    gang.barrier()

    # Broadcast Python objects
    gang.broadcast_objects(objects, source_rank=0)

Parallel Training Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In fairseq2, parallel training is organized around Data Parallel (DP) Gangs and Tensor Parallel (TP) Gangs, which together enable scalable training of large models.
For example, the ``setup_parallel_gangs(root_gang, tp_size=2)`` function creates a root gang (e.g., 8 processes) and then creates 2 DP gangs and 4 TP gangs.

fairseq2 also supports hybrid-sharding FSDP configurations through ``setup_hybrid_fsdp_gangs()``, which creates specialized gang arrangements for efficient model sharding and replication across devices.

.. image:: /_static/img/gang.svg
    :width: 600px
    :align: center
    :alt: Gang Architecture

Structure and Organization of DP and TP Gangs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Data Parallel (DP) Gangs:

   - Group GPUs that process different data batches (or parts of batches).
   - Synchronize gradients across the GPUs in the same DP Gang after the backward pass.
   - Example: DP Gang 1 has GPUs 0, 2, 4, and 6, while DP Gang 2 has GPUs 1, 3, 5, and 7.

2. Tensor Parallel (TP) Gangs:

   - Group GPUs that split the model parameters for parallel computation.
   - Operate within the same DP Gang but compute sequentially during forward and backward passes.
   - Example: TP Gang 1 has GPUs 0 and 1, while TP Gang 2 has GPUs 2 and 3.

**A Single Training Step**

1. Forward Pass:

    - Input data is distributed among Data Parallel (DP) Gangs.
    - Each Tensor Parallel (TP) Gang processes its segment of the model sequentially, transferring activations between GPUs.

2. Backward Pass:

    - Gradients are calculated in the reverse sequence of the forward pass within TP Gangs.
    - Activation gradients are relayed back to preceding GPUs.

3. Gradient Synchronization:

    - Gradients are synchronized across all GPUs within each DP Gang.

4. Parameter Update:

    - Each GPU updates its local parameters (or shards, if utilizing TP).

.. dropdown:: How step-by-step parallel training works
    :icon: code
    :animate: fade-in

    - Step 1: Data Splitting
        - The global input batch is divided into sub-batches, each assigned to a specific DP Gang

    - Step 2: Forward Pass (TP Gangs)
        - Each TP Gang processes its shard of the model sequentially:
        - GPU 0 (TP Gang 1) computes layers 0-2, passing activations to GPU 1.
        - GPU 1 (TP Gang 1) computes layers 3-5 using these activations.
        - This process is repeated for all TP Gangs.

    - Step 3: Backward Pass (TP Gangs)
        - The reverse order of the forward pass:
        - Gradients of layers 2-3 are computed on GPU 1.
        - Activation gradients are sent back to GPU 0, which computes gradients for layers 0-1.

    - Step 4: Gradient Synchronization (DP Gangs)
        - Gradients are synchronized across GPUs within the same DP Gang using an ``all_reduce`` operation.

    - Step 5: Parameter Update
        - Each GPU updates its parameters or model shards locally after synchronization.

.. dropdown:: The list of environment variables picked up by fairseq2
    :icon: code
    :animate: fade-in

    The following environment variables control distributed training:

    - ``WORLD_SIZE``: Total number of processes.
    - ``RANK``: Rank of the current process.
    - ``LOCAL_WORLD_SIZE``: Number of processes per node.
    - ``LOCAL_RANK``: Local rank within a node.
    - ``MASTER_ADDR``: Address of rank 0 process
    - ``MASTER_PORT``: Port for rank 0 process

    ``torchrun`` and SLURM automatically sets these variables.


Usage Examples
--------------

1. Basic Gang Setup
^^^^^^^^^^^^^^^^^^^

For standard distributed training:

.. code-block:: python

    from fairseq2.gang import setup_root_gang
    from datetime import timedelta

    # Initialize the default gang with custom settings
    gang = setup_root_gang(
        timeout=timedelta(minutes=30),  # Custom timeout for monitored barriers
        monitored=True  # Enable monitored barriers for deadlock detection
    )

    print(f"Process rank: {gang.rank}, World size: {gang.size}")


.. note::

    If running locally (no ``torch.distributed`` backend), a ``FakeGang`` is created.
    This is useful for local testing and debugging.

    If running in a distributed environment, a ``ProcessGroupGang`` is created.

2. Create a Sub-Gang
^^^^^^^^^^^^^^^^^^^^^

You can create sub-groups of processes (`e.g.`, for model parallelism):

.. code-block:: python

    sub_gang = gang.make_gang([0, 1, 2])
    if sub_gang:
        print(f"Sub-gang rank: {sub_gang.rank}, Size: {sub_gang.size}")

3. Data & Tensor Parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from fairseq2.gang import setup_parallel_gangs

    # Setup root gang first
    root_gang = setup_root_gang()

    # Create DP and TP gangs with tensor parallel size = 2
    gangs = setup_parallel_gangs(root_gang, tp_size=2)

    print(f"Data Parallel Rank: {gangs.dp.rank}")
    print(f"Tensor Parallel Rank: {gangs.tp.rank}")


4. Collective Operations
^^^^^^^^^^^^^^^^^^^^^^^^

A minimal example of distributed training with gangs:

.. code-block:: python

    # script.py
    import torch
    from fairseq2.gang import setup_root_gang, ReduceOperation

    # Initialize gang
    gang = setup_root_gang()

    # Dummy tensor
    tensor = torch.tensor(gang.rank + 1.0, device=gang.device)

    # Sum tensor across all processes
    gang.all_reduce(tensor, ReduceOperation.SUM)
    print(f"Rank {gang.rank}: Tensor after all_reduce = {tensor.item()}")

    # Synchronize
    gang.barrier()


To run this example w/ torchrun:

.. code-block:: bash

    torchrun --nproc_per_node=4 script.py


Best Practices
--------------

1. **Development Workflow**

   - Start with ``FakeGang`` for local development
   - Move to distributed training once code works locally
   - Use monitored barriers to detect deadlocks

2. **Process Layout**

   - Place adjacent ranks on same node for TP efficiency
   - Balance DP and TP sizes based on model and data characteristics

3. **Launching Jobs**

   - Use ``torchrun`` for simple distributed training:

     .. code-block:: bash

        torchrun --nproc_per_node=4 train.py

   - Use SLURM for cluster environments:

     .. code-block:: bash

        srun -N 1 --gres=gpu:4 --cpus-per-task=12 python train.py


4. **Error Handling**

   - Always synchronize processes with barriers at critical points
   - Monitor for process failures in production settings
   - Enable logging for debugging distributed issues

5. **Device Placement**

   - Ensure tensors are on correct devices before collective ops
   - Use ``gang.device`` to get the appropriate device

6. **Resource Management**

   - Close gangs properly when done

See Also
--------

- :ref:`basics-trainer` - How Gang integrates with training
