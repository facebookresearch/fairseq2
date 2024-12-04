.. _recipes-lm-convert-checkpoint:

===================
Convert Checkpoints
===================

.. module:: fairseq2.recipes.llama.convert_checkpoint

The checkpoint conversion module provides utilities to convert fairseq2 model checkpoints to different formats for interoperability with other frameworks.

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

For the complete list of architectures and their configurations, see :module:`fairseq2.models.llama.archs`.

Output Format
-------------

The converter produces:

1. Model weights in the reference format:
   - Single checkpoint: ``consolidated.00.pth``
   - Sharded checkpoints: ``consolidated.{00,01,02...}.pth``

2. ``params.json`` containing model configuration:

.. code-block:: json

    {
        "model": {
            "dim": 2048,              # Model dimension
            "n_layers": 16,           # Number of layers
            "n_heads": 32,            # Number of attention heads
            "n_kv_heads": 8,          # Number of key/value heads (if different from n_heads)
            "multiple_of": 256,       # FFN dimension multiple
            "ffn_dim_multiplier": 1.5,# FFN dimension multiplier (if not 1.0)
            "rope_theta": 500000.0,   # RoPE theta value
            "norm_eps": 1e-5          # Layer norm epsilon
        }
    }

Usage Example
-------------

1. Convert a fairseq2 checkpoint to reference format:

.. code-block:: bash

    fairseq2 llama convert_checkpoint --model llama3_2_1b \
        /path/to/fairseq2/checkpoint \
        /path/to/output/dir

2. Convert to HuggingFace format:

After converting to reference format, use the HuggingFace conversion script to convert to HF format:

.. code-block:: bash

    python -m transformers.models.llama.convert_llama_weights_to_hf \
        --input_dir /path/to/output/dir \
        --model_size 1B \
        --output_dir /path/to/hf/model

See Also
--------

- :doc:`End-to-End Fine-Tuning Tutorial </tutorials/end_to_end_fine_tuning>`
- :class:`fairseq2.models.llama.LLaMAConfig`
- :class:`fairseq2.models.llama.archs`