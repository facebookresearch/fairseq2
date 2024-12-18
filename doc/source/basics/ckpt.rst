.. _basics-ckpt-management:

:octicon:`check-circle` Checkpoint Management
=============================================

The checkpoint manager in fairseq2 handles saving and loading of model states, optimizer states, and training progress.
It provides a robust way to:

- Save model checkpoints during training
- Load checkpoints to resume training
- Manage multiple checkpoints with policies like keeping N-best or last N checkpoints
- Handle distributed training scenarios including FSDP (Fully Sharded Data Parallel)

Architecture Overview
---------------------

.. mermaid::

    graph TD
        A[Trainer] -->|uses| B[CheckpointManager]
        B -->|saves| C[Model State]
        B -->|saves| D[Optimizer State] 
        B -->|saves| E[Training Metadata]
        B -->|manages| F[Checkpoint Files]
        G[Model Loader] -->|loads| B

Basic Usage
-----------

Saving Checkpoints
^^^^^^^^^^^^^^^^^^

The :class:`fairseq2.checkpoint.manager.CheckpointManager` provides a transactional API for saving checkpoints:

.. code-block:: python

    # Initialize checkpoint manager
    ckpt_manager = FileCheckpointManager(
        checkpoint_dir=Path("checkpoints"),
        gang=root_gang  # For distributed training coordination
    )

    # Begin checkpoint operation
    ckpt_manager.begin_checkpoint(step_nr=1000)

    # Save model and optimizer state
    ckpt_manager.save_state({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step_nr": 1000,
        "epoch": 5
    })

    # Save validation score if needed
    ckpt_manager.save_score(valid_score)

    # Commit the checkpoint
    ckpt_manager.commit_checkpoint()

Loading Checkpoints
^^^^^^^^^^^^^^^^^^^

To load the latest checkpoint:

.. code-block:: python

    try:
        # Load the last checkpoint
        step_nr, state = ckpt_manager.load_last_checkpoint()
        
        # Restore model and optimizer state
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        
        print(f"Restored checkpoint from step {step_nr}")
    except CheckpointNotFoundError:
        print("No checkpoint found, starting fresh")

Checkpoint Management Policies
------------------------------

The :class:`fairseq2.checkpoint.manager.CheckpointManager` supports different policies for managing multiple checkpoints:

Keep Last N Checkpoints
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Keep only the last 5 checkpoints
    ckpt_manager.keep_last_n_checkpoints(n=5)

Keep Best N Checkpoints
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Keep the 3 checkpoints with best validation scores
    ckpt_manager.keep_best_n_checkpoints(
        n=3,
        lower_better=True  # True if lower scores are better
    )

Distributed Training Support
----------------------------

The :class:`fairseq2.checkpoint.manager.CheckpointManager` handles distributed training scenarios including:

- Data Parallel (DP) training
- Fully Sharded Data Parallel (FSDP) training
- Tensor Parallel (TP) training

For FSDP, the manager provides special handling:

.. code-block:: python

    # Save consolidated (non-sharded) model state
    ckpt_manager.save_consolidated_fsdp_model(model)

Checkpoint Structure
--------------------

A checkpoint directory contains:

.. code-block:: text

    checkpoint_dir/
    ├── model.yaml           # Model metadata
    └── step_1000/          # Checkpoint at step 1000
        └── model.pt        # Model training state

For sharded checkpoints (FSDP), each rank has its own files:

.. code-block:: text

    checkpoint_dir/
    ├── model.yaml           # Model metadata
    ├── step_1000/
    |   ├── model.pt         # Consolidated model
    |   ├── rank_0.pt        # Model rank 0 state
    |   └── rank_1.pt        # Model rank 1 state

Error Handling
--------------

The checkpoint system provides specific exceptions for error cases:

- ``CheckpointError``: Base class for checkpoint-related errors
- ``CheckpointNotFoundError``: Raised when attempting to load non-existent checkpoint
- ``InvalidOperationError``: Raised for invalid checkpoint operations

Example error handling:

.. code-block:: python

    try:
        ckpt_manager.load_checkpoint(step_nr=1000)
    except CheckpointNotFoundError:
        print("Checkpoint not found")
    except CheckpointError as e:
        print(f"Error loading checkpoint: {e}")

Best Practices
--------------

1. Always use the transactional API (``begin_checkpoint``/``commit_checkpoint``) to ensure checkpoint consistency

2. Implement checkpoint cleanup policies to manage storage space

3. Include sufficient metadata in checkpoints for reproducibility

4. Handle checkpoint errors gracefully in production code

5. For distributed training, ensure proper gang coordination
