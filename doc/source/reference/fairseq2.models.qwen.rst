.. _api-models-qwen:

====================
fairseq2.models.qwen
====================

.. currentmodule:: fairseq2.models.qwen

The Qwen module provides support for Qwen2.5 and Qwen3 language models.
It includes model configurations, hub access, tokenizers, and utilities for loading and working with Qwen models.

Quick Start
-----------

.. code-block:: python

    from fairseq2.models.qwen import get_qwen_model_hub, get_qwen_tokenizer_hub

    # Get the model hub
    hub = get_qwen_model_hub()

    # List available architectures
    print("Available Qwen architectures:")
    for arch in sorted(hub.get_archs()):
        print(f"  - {arch}")

    # Load a model
    model = hub.load_model("qwen25_7b")

    # Load corresponding tokenizer
    tokenizer = get_qwen_tokenizer_hub().load_tokenizer("qwen25_7b")

    # Generate some text
    text = "The future of AI is"
    encoder = tokenizer.create_encoder()
    encoded = encoder(text)
    # ... model inference code ...

Available Models
----------------

The Qwen family includes several model sizes and versions:

**Qwen 2.5 Series:**
- ``qwen25_1_5b`` - 1.5B parameters
- ``qwen25_3b`` - 3B parameters
- ``qwen25_7b`` - 7B parameters
- ``qwen25_14b`` - 14B parameters
- ``qwen25_32b`` - 32B parameters

**Qwen 3 Series:**
- ``qwen3_0.6b`` - 0.6B parameters
- ``qwen3_1.7b`` - 1.7B parameters
- ``qwen3_4b`` - 4B parameters
- ``qwen3_8b`` - 8B parameters
- ``qwen3_14b`` - 14B parameters
- ``qwen3_32b`` - 32B parameters

Model Configuration
-------------------

QwenConfig
~~~~~~~~~~

.. autoclass:: QwenConfig
    :members:
    :show-inheritance:

    Configuration class for Qwen models. Defines the architecture parameters such as
    model dimensions, number of layers, attention heads, and other architectural choices.

    **Key Parameters:**

    * ``model_dim`` - The dimensionality of the model (default: 3584)
    * ``num_layers`` - Number of decoder layers (default: 28)
    * ``num_attn_heads`` - Number of attention heads (default: 28)
    * ``num_key_value_heads`` - Number of key/value heads for GQA (default: 4)
    * ``max_seq_len`` - Maximum sequence length (default: 32,768)
    * ``vocab_size`` - Vocabulary size (default: 152,064)

    **Example:**

    .. code-block:: python

        from fairseq2.models.qwen import QwenConfig

        # Create custom configuration
        config = QwenConfig()
        config.model_dim = 4096
        config.num_layers = 32
        config.num_attn_heads = 32
        config.max_seq_len = 16384

        # Or get pre-defined architecture
        from fairseq2.models.qwen import get_qwen_model_hub
        hub = get_qwen_model_hub()
        config = hub.get_arch_config("qwen25_7b")


Model Factory
-------------

QwenFactory
~~~~~~~~~~~

.. autoclass:: QwenFactory
    :members:
    :show-inheritance:

    Factory class for creating Qwen models. Handles model instantiation and checkpoint loading.

create_qwen_model
~~~~~~~~~~~~~~~~~

.. autofunction:: create_qwen_model

    Creates a Qwen model instance with the specified configuration.

    .. code-block:: python

        from fairseq2.models.qwen import create_qwen_model, QwenConfig

        config = QwenConfig()
        config.model_dim = 2048
        config.num_layers = 24

        model = create_qwen_model(config)

Tokenizer
---------

QwenTokenizer
~~~~~~~~~~~~~

.. autoclass:: QwenTokenizer
    :members:
    :show-inheritance:

    Tokenizer for Qwen models. Handles text encoding and decoding using the Qwen vocabulary.

QwenTokenizerConfig
~~~~~~~~~~~~~~~~~~~

.. autoclass:: QwenTokenizerConfig
    :members:
    :show-inheritance:

    Configuration for the Qwen tokenizer.

get_qwen_tokenizer_hub
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: get_qwen_tokenizer_hub

    Returns the tokenizer hub for Qwen tokenizers.

    .. code-block:: python

        from fairseq2.models.qwen import get_qwen_tokenizer_hub

        tokenizer_hub = get_qwen_tokenizer_hub()

        # Load tokenizer through hub
        tokenizer = tokenizer_hub.load_tokenizer("qwen25_7b")

Interoperability
----------------

convert_qwen_state_dict
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: convert_qwen_state_dict

    Converts Qwen model state dictionaries between different formats (e.g., from Hugging Face format).

    .. code-block:: python

        from fairseq2.models.qwen import convert_qwen_state_dict
        import torch

        # Load checkpoint from Hugging Face format
        hf_state_dict = torch.load("qwen_hf_checkpoint.pt")

        # Convert to fairseq2 format
        fs2_state_dict = convert_qwen_state_dict(hf_state_dict)

export_qwen
~~~~~~~~~~~

.. autofunction:: export_qwen

    Exports fairseq2 Qwen models to other formats for interoperability.

Sharding
--------

get_qwen_shard_specs
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: get_qwen_shard_specs

    Returns sharding specifications for distributed training and inference of Qwen models.

    .. code-block:: python

        from fairseq2.models.qwen import get_qwen_shard_specs, QwenConfig

        config = QwenConfig()
        shard_specs = get_qwen_shard_specs(config)

Constants
---------

QWEN_FAMILY
~~~~~~~~~~~

.. autodata:: QWEN_FAMILY
    :annotation: = "qwen"

    The family name identifier for Qwen models.

Complete Examples
-----------------

Basic Model Usage
~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch

    from fairseq2.models.qwen import get_qwen_model_hub, get_qwen_tokenizer_hub
    from fairseq2.device import get_default_device
    from fairseq2.nn import BatchLayout

    device = get_default_device()

    # Load model and tokenizer
    hub = get_qwen_model_hub()
    model = hub.load_model("qwen25_7b", device=device)
    tokenizer = get_qwen_tokenizer_hub().load_tokenizer("qwen25_7b")

    # Prepare input
    texts = ["The capital of France is", "The capital of Germany is"]
    encoder = tokenizer.create_encoder()
    tokens = torch.vstack([encoder(text) for text in texts]).to(device)

    # Run inference (simplified)
    model.eval()
    with torch.inference_mode():
        seqs_layout = BatchLayout.of(tokens)
        output = model(tokens, seqs_layout=seqs_layout)
        # Process output...

Custom Architecture
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from fairseq2.models.qwen import get_qwen_model_hub, QwenConfig

    hub = get_qwen_model_hub()

    # Get base configuration and modify
    config = hub.get_arch_config("qwen25_7b")
    config.max_seq_len = 16384  # Reduce sequence length
    config.dropout_p = 0.1      # Add dropout

    # Create model with custom config
    model = hub.create_new_model(config)

Loading from Custom Checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pathlib import Path
    from fairseq2.models.qwen import get_qwen_model_hub

    hub = get_qwen_model_hub()
    config = hub.get_arch_config("qwen25_7b")

    # Load from custom checkpoint
    checkpoint_path = Path("/path/to/my/qwen_checkpoint.pt")
    model = hub.load_custom_model(checkpoint_path, config)

Architecture Comparison
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from fairseq2.models.qwen import get_qwen_model_hub

    hub = get_qwen_model_hub()

    # Compare different Qwen architectures
    architectures = ["qwen25_3b", "qwen25_7b", "qwen25_14b"]

    for arch in architectures:
        config = hub.get_arch_config(arch)
        params = config.model_dim * config.num_layers * config.num_attn_heads
        print(f"{arch}:")
        print(f"  Model dim: {config.model_dim}")
        print(f"  Layers: {config.num_layers}")
        print(f"  Attention heads: {config.num_attn_heads}")
        print(f"  Approx parameters: ~{params//1_000_000}M")
        print()

See Also
--------

* :doc:`/reference/fairseq2.models.hub` - Model hub API reference
* :doc:`/guides/add_model` - Tutorial on adding new models
* :doc:`/basics/assets` - Understanding the asset system
