.. _tutorial-add-model:

:octicon:`ruby` Add Your Own Model
==================================

.. dropdown:: What you will learn
    :icon: multi-select
    :animate: fade-in

    * How to add a new model to an existing model family
    * How to register model configurations (architectures)
    * How to create asset cards for your models
    * How to verify your model integration works correctly

.. dropdown:: Prerequisites
    :icon: multi-select
    :animate: fade-in

    * Get familiar with fairseq2 basics (:ref:`basics-design-philosophy`)
    * Understanding of fairseq2 assets system (:ref:`basics-assets`)
    * Ensure you have fairseq2 installed

Overview
--------

fairseq2 uses a model system that makes it easy to add support for new models.
There are two main scenarios:

1. **Adding a new model to an existing family** (most common) - When you want to add a new size or variant of an existing model architecture
2. **Creating an entirely new model family** (advanced) - When you need to implement a completely new model architecture

This tutorial focuses on the first scenario, which covers 95% of use cases. For the second scenario, refer to the existing model family implementations as reference.

Understanding Model Families
----------------------------

In fairseq2, a **model family** groups related model architectures that share the same underlying implementation but differ in size or configuration. For example:

- **Qwen family**: ``qwen25_3b``, ``qwen25_3b``, ``qwen25_14b``, ``qwen3_8b``, etc.
- **LLaMA family**: ``llama3_8b``, ``llama3_70b``, ``llama3_2_1b``, etc.
- **Mistral family**: ``mistral_7b``, ``mistral_8x7b``, etc.

Each family consists of:

- **Model configurations** (architectures): Define structural parameters (layers, dimensions, etc.)
- **Asset cards**: YAML files specifying download locations and metadata
- **Model implementation**: The actual PyTorch model code and loading logic
- **Model hub**: A unified interface providing methods to work with the family

Working with Model Hubs
^^^^^^^^^^^^^^^^^^^^^^^^

Each model family provides a hub that exposes advanced functionality beyond simple model loading:

.. code-block:: python

    from fairseq2.models.qwen import get_qwen_model_hub

    # Get the model hub for Qwen family
    hub = get_qwen_model_hub()

    # List available architectures
    archs = hub.get_archs()
    print(f"Available Qwen architectures: {archs}")

    # Get architecture configuration
    config = hub.get_arch_config("qwen3_0.6b")

    # Create a newly initialized model (random weights)
    new_model = hub.create_new_model(config)

    # Load model from asset card
    model = hub.load_model("qwen3_0.6b")

    # Load model from custom checkpoint
    from pathlib import Path
    custom_model = hub.load_custom_model(Path("/path/to/checkpoint.pt"), config)

For detailed information on all hub capabilities, see :doc:`/reference/models/hub`.

Step-by-Step Guide: Adding a Model to Existing Family
-----------------------------------------------------

Let's walk through adding ``qwen25_3b_instruct`` to the existing Qwen family.

Step 1: Add Model Architecture Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Model architectures are defined as configuration presets that specify the structural parameters of your model.

1. **Navigate to the model family's config file:**

   .. code-block:: bash

       src/fairseq2/models/qwen/config.py

2. **Add a new architecture function:**

   .. code-block:: python

       @arch("qwen25_3b_instruct")
       def qwen25_3b_instruct() -> QwenConfig:
           """Configuration for Qwen2.5-3B-Instruct model.
           """
           config = QwenConfig()

           # Set model dimensions and structure
           config.model_dim = 2048
           ...

           return config

   **Key points:**

   - The ``@arch`` decorator registers the configuration with the given name
   - The function name should match or describe the architecture
   - Use the appropriate config class for the model family (``QwenConfig`` for Qwen models)
   - Set all necessary parameters for your specific model variant

Step 2: Create Asset Card
^^^^^^^^^^^^^^^^^^^^^^^^^

Asset cards are YAML files that tell fairseq2 where to find your model checkpoints and how to load them.

1. **Navigate to the model family's asset card file:**

    .. code-block:: bash

        src/fairseq2/assets/cards/models/qwen.yaml

2. **Add a new asset card entry:**

    .. code-block:: yaml

        name: qwen25_3b_instruct
        model_family: qwen
        model_arch: qwen25_3b
        checkpoint: "hg://qwen/qwen2.5-3b-instruct"
        tokenizer: "hg://qwen/qwen2.5-3b-instruct"
        tokenizer_family: qwen
        tokenizer_config:
            use_im_end: true

- ``name``: The model name users will use (e.g., ``load_model("qwen25_3b_instruct")``)
- ``model_family``: Which model family handles this model (``qwen``)
- ``model_arch``: Which architecture configuration to use (``qwen25_3b``)
- ``checkpoint``: Where to download the model weights from
- ``tokenizer``: Where to download the tokenizer from
- ``tokenizer_family``: Which tokenizer family to use
- ``tokenizer_config``: Tokenizer-specific settings

For more details on asset card options, see :ref:`basics-assets`.

Step 3: Verify the Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After adding the configuration and asset card, verify that your model is properly registered:

1. **Check if model is recognized:**

   .. code-block:: bash

       # List all models to see if yours appears
       python -m fairseq2.assets list --kind model

       # Look specifically for your model
       python -m fairseq2.assets list --kind model | grep qwen25_3b_instruct

2. **Test model loading:**

   .. code-block:: python

       import fairseq2
       from fairseq2.models.hub import load_model

       # Test loading your model
       try:
           model = load_model("qwen25_3b_instruct")
           print(f"✓ Success! Loaded model with {sum(p.numel() for p in model.parameters())} parameters")
       except Exception as e:
           print(f"✗ Error: {e}")

3. **Inspect model metadata:**

   .. code-block:: bash

       # Show detailed model information
       python -m fairseq2.assets show qwen25_3b_instruct

Asset Source Options
--------------------

fairseq2 supports multiple sources for model checkpoints and tokenizers:

Hugging Face Hub (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most common and convenient option:

.. code-block:: yaml

    checkpoint: "hg://qwen/qwen2.5-3b-instruct"
    tokenizer: "hg://qwen/qwen2.5-3b-instruct"

Note that only safetensors are supported for checkpoints.

Local Files
^^^^^^^^^^^

For development or custom models:

.. code-block:: yaml

    checkpoint: "file:///path/to/my/model.pt"
    tokenizer: "file:///path/to/my/tokenizer"

HTTP URLs
^^^^^^^^^

Direct download links:

.. code-block:: yaml

    checkpoint: "https://example.com/models/my_model.pt"


Common Model Parameters
-----------------------

When creating new architecture configurations, here are the most common parameter naming conventions you'll find in fairseq2 (it may vary depending on the model architecture):

Core Architecture
^^^^^^^^^^^^^^^^^

.. code-block:: python

    config.model_dim = 2048           # Model dimensionality
    config.num_layers = 36            # Number of transformer layers
    config.num_attn_heads = 16        # Number of attention heads
    config.num_key_value_heads = 2    # Key/value heads (for GQA/MQA)
    config.ffn_inner_dim = 11_008     # Feed-forward network inner dimension

Vocabulary & Sequence
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    config.vocab_size = 151_936       # Vocabulary size
    config.max_seq_len = 32_768       # Maximum sequence length
    config.tied_embeddings = True     # Tie input/output embeddings

Training & Architecture Details
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    config.head_dim = 128             # Attention head dimension (optional)
    config.qkv_proj_bias = False      # Query/key/value projection bias
    config.dropout_p = 0.0            # Dropout probability
    config.rope_theta = 1_000_000.0   # RoPE theta parameter

Troubleshooting
---------------

Model Not Found Error
^^^^^^^^^^^^^^^^^^^^^

If you get ``ModelNotKnownError``:

1. **Check asset card syntax:** Ensure your YAML is valid
2. **Verify names match:** Asset card ``name`` should match what you're requesting
3. **Check architecture registration:** Ensure the ``@arch`` decorated function exists
4. **Restart Python:** Changes to config files require restarting your Python session

Architecture Configuration Error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you get architecture-related errors:

1. **Verify decorator:** Ensure ``@arch("name")`` is properly applied
2. **Check architecture name:** Asset card ``model_arch`` must match the registered name
3. **Validate parameters:** Ensure all required config parameters are set

Download/Loading Errors
^^^^^^^^^^^^^^^^^^^^^^^

If model download or loading fails:

1. **Check URLs:** Verify checkpoint and tokenizer URLs are accessible
2. **Test connectivity:** Ensure you have internet access and proper authentication
3. **Check file paths:** For local files, verify paths exist and are readable
4. **Validate checkpoint format:** Ensure checkpoint is compatible with the model family

Configuration Validation Errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you get validation errors:

1. **Check parameter types:** Ensure integers are integers, strings are strings, etc.
2. **Validate ranges:** Some parameters may have valid ranges (e.g., positive integers)
3. **Review dependencies:** Some parameters may depend on others (e.g., head dimensions)


Example: Complete Implementation
--------------------------------

Here's a complete example showing all the files you need to modify to add ``qwen25_3b_instruct``:

**1. Architecture Configuration** (``src/fairseq2/models/qwen/config.py``):

.. code-block:: python

    @arch("qwen25_3b_instruct")
    def qwen25_3b_instruct() -> QwenConfig:
        """Qwen2.5-3B-Instruct: Language model with 3B parameters.

        Paper: https://arxiv.org/abs/2024.xxxxx
        """
        config = QwenConfig()

        config.model_dim = 2048
        ...

        return config

**2. Asset Card** (``src/fairseq2/assets/cards/models/qwen.yaml``):

.. code-block:: yaml

    ---

    name: qwen25_3b_instruct
    model_family: qwen
    model_arch: qwen25_3b
    checkpoint: "hg://qwen/qwen2.5-3b-instruct"
    tokenizer: "hg://qwen/qwen2.5-3b-instruct"
    tokenizer_family: qwen
    tokenizer_config:
      use_im_end: true

**3. Command Line Verification**:

.. code-block:: bash

    # Check model is listed
    python -m fairseq2.assets list --kind model | grep qwen25_3b_instruct

    # Show model details
    python -m fairseq2.assets show qwen25_3b_instruct

    # Quick load test
    python -c "
    from fairseq2.models.hub import load_model
    model = load_model('qwen25_3b_instruct')
    print('✓ Success!')
    "

This complete example shows all the steps needed to add a new model to fairseq2.
The process is straightforward but requires attention to detail to ensure all components work together correctly.
