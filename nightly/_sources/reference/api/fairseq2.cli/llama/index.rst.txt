.. _reference-cli-llama:

===================
Convert Checkpoints
===================

.. module:: fairseq2.cli.commands.llama

The checkpoint conversion handlers provides utilities to convert fairseq2 model checkpoints to different formats for interoperability with other frameworks.

Command Line Interface
----------------------

.. code-block:: bash

    fairseq2 llama convert_checkpoint --model <architecture> <fairseq2_checkpoint_dir> <output_dir>

Arguments
^^^^^^^^^

- ``--model <architecture>``: The model architecture name (e.g., ``llama3_2_1b``) to generate correct ``params.json``
- ``<fairseq2_checkpoint_dir>``: Directory containing the fairseq2 checkpoint (model.pt or model.{0,1,2...}.pt for sharded checkpoints)
- ``<output_dir>``: Output directory to store the converted checkpoint

Supported Architectures
-----------------------

The converter supports various LLaMA architectures including:

- LLaMA 1: 7B, 13B, 33B, 65B
- LLaMA 2: 7B, 13B, 70B
- LLaMA 3: 8B, 70B
- LLaMA 3.1: 8B, 70B
- LLaMA 3.2: 1B, 3B

For the complete list of architectures and their configurations, see :mod:`fairseq2.models.llama.archs`.

Output Format
-------------

The converter produces model weights in the reference format:
   - Single checkpoint: ``consolidated.00.pth``
   - Sharded checkpoints: ``consolidated.{00,01,02...}.pth``

Usage Example
-------------

1. Convert a fairseq2 checkpoint to reference format:

.. code-block:: bash

    fairseq2 llama convert_checkpoint --model llama3_2_1b \
        /path/to/fairseq2/checkpoint \
        /path/to/output/dir

.. note::

    Architecture ``--model`` must exist and be defined in `e.g.` :meth:`fairseq2.models.llama._config.register_llama_configs`.

API Details
-----------

.. autoclass:: ConvertLLaMACheckpointHandler

See Also
--------

- :doc:`End-to-End Fine-Tuning Tutorial </tutorials/end_to_end_fine_tuning>`
- :class:`fairseq2.models.llama._config.LLaMAConfig`
