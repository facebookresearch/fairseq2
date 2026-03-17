==================================================
HuggingFace Model SFT Training with fairseq2
==================================================

.. currentmodule:: fairseq2.models.hg_qwen_omni

fairseq2 provides integration with HuggingFace Transformers models through the
:mod:`fairseq2.models.hg_qwen_omni` module. This guide walks through a complete
supervised fine-tuning (SFT) example using a Gemma model loaded via HuggingFace
and trained with fairseq2's distributed training infrastructure.

The full example script is located at ``examples/simple_gemma_sft.py``.

Prerequisites
=============

Ensure fairseq2 is installed and the virtual environment is activated:

.. code:: bash

    source .venv/bin/activate

The example uses ``google/gemma-3-1b-it`` by default. HuggingFace will
download the model automatically on first use.

Running the Example
===================

**CPU mode** (slow, for testing only):

.. code:: bash

    python examples/simple_gemma_sft.py --device cpu

**Single GPU mode** (recommended):

.. code:: bash

    python examples/simple_gemma_sft.py --device cuda

**Multi-GPU mode with FSDP** (requires 2+ GPUs):

.. code:: bash

    torchrun --nproc_per_node=2 examples/simple_gemma_sft.py --device cuda

**Custom training parameters**:

.. code:: bash

    python examples/simple_gemma_sft.py --device cuda --num-epochs 2 --learning-rate 1e-5 --batch-size 8

What the Example Demonstrates
==============================

The script covers several key fairseq2 features:

1. **Loading a HuggingFace model** using :func:`load_causal_lm` from
   fairseq2's HG integration module.

2. **Distributed training** using fairseq2's :class:`~fairseq2.gang.Gang`
   abstraction. When launched with ``torchrun``, the script automatically
   detects the multi-GPU environment and initializes
   :class:`~fairseq2.gang.ProcessGroupGang`.

3. **FSDP sharding** using :func:`~fairseq2.nn.fsdp.to_fsdp2` and
   :func:`apply_fsdp_to_hg_transformer_lm` for memory-efficient multi-GPU
   training with layer-wise sharding granularity.

4. **Checkpoint management** using
   :class:`~fairseq2.checkpoint.StandardCheckpointManager` for saving and
   automatically resuming training from the last checkpoint.

5. **Tokenization** using :func:`load_hg_tokenizer_simple` to load the
   HuggingFace tokenizer with proper special token handling.

Key Code Patterns
=================

Loading a HuggingFace Model
---------------------------

.. code:: python

    from fairseq2.models.hg_qwen_omni import load_causal_lm, load_hg_tokenizer_simple

    model = load_causal_lm("google/gemma-3-1b-it", device=device)
    tokenizer = load_hg_tokenizer_simple("google/gemma-3-1b-it")

Distributed Setup with Gang
----------------------------

The example uses fairseq2's :class:`~fairseq2.gang.Gang` rather than raw
``torch.distributed`` calls:

.. code:: python

    from fairseq2.gang import ProcessGroupGang, create_parallel_gangs, create_fsdp_gangs

    gang = ProcessGroupGang.create_default_process_group(device)
    gangs = create_parallel_gangs(gang)
    gangs = create_fsdp_gangs(gangs)

See :doc:`/concepts/gang` for a detailed explanation of the gang abstraction.

Applying FSDP
-------------

.. code:: python

    from fairseq2.nn.fsdp import to_fsdp2
    from fairseq2.models.hg_qwen_omni import apply_fsdp_to_hg_transformer_lm

    # Mark layer boundaries for FSDP sharding
    apply_fsdp_to_hg_transformer_lm(model)

    # Wrap the model with FSDP
    to_fsdp2(model, gangs)

Checkpoint Management
---------------------

.. code:: python

    from fairseq2.checkpoint import StandardCheckpointManager

    manager = StandardCheckpointManager(output_dir / "checkpoints", ...)

    # Save checkpoint
    manager.save(step, state_dict)

    # Resume from last checkpoint (returns None if no checkpoint exists)
    last_step = manager.get_last_step()

Checkpointing Features
=======================

The script uses fairseq2's :class:`~fairseq2.checkpoint.StandardCheckpointManager`
for robust checkpointing:

- **Automatic resume**: If a checkpoint exists in ``--output-dir``, training
  automatically resumes from the last step.
- **Saves model and optimizer state** for proper resumption.
- **Distributed-safe**: Works correctly in multi-GPU setups.
- **Checkpoints saved to**: ``{output_dir}/checkpoints/step_{N}/``

.. code:: bash
    :caption: Example - Resume Training

    # First run (trains and saves checkpoint)
    python examples/simple_gemma_sft.py --output-dir my_run --num-epochs 1

    # Resume from checkpoint (automatically detects and loads)
    python examples/simple_gemma_sft.py --output-dir my_run --num-epochs 2

Multi-GPU Notes
===============

When running with FSDP:

- Each rank sees its local device (e.g., ``cuda:0`` from rank 0's perspective).
  The model is actually sharded across all GPUs — this is expected FSDP behavior.
- The script skips the generation test when FSDP is enabled because
  FSDP-wrapped models require all ranks to participate in forward passes.
- Only rank 0 prints output to avoid clutter.

To disable FSDP in multi-GPU mode:

.. code:: bash

    torchrun --nproc_per_node=2 examples/simple_gemma_sft.py --device cuda --no-fsdp

Training Stability
==================

The script uses conservative settings for stable training:

- **fp32 precision** instead of fp16 (more stable without loss scaling)
- **Low learning rate** (5e-6) to prevent instability
- **Gradient clipping** (max norm 1.0) to prevent exploding gradients
- **NaN detection** with early stopping if training becomes unstable

.. note::

    This is an educational example demonstrating the basics. For production
    training, use fairseq2's full :class:`~fairseq2.trainer.Trainer` class with
    proper data pipelines, validation, and early stopping.
