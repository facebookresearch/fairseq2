.. _api-models-hub:

===================
fairseq2.models.hub
===================

.. currentmodule:: fairseq2.models.hub

The model hub provides a unified interface for working with model families in fairseq2.
Each model family has its own hub that exposes methods for loading models, creating new instances,
listing architectures, and more.

Quick Start
-----------

.. code-block:: python

    from fairseq2.models.qwen import get_qwen_model_hub

    # Get the model hub for Qwen family
    hub = get_qwen_model_hub()

    # List available architectures
    archs = hub.get_archs()
    print(f"Available architectures: {archs}")

    # Create a new uninitialized model
    config = hub.get_arch_config("qwen25_7b")
    model = hub.create_new_model(config)

    # Load a model from asset card
    model = hub.load_model("qwen25_7b")

    # Load a model from custom checkpoint
    from pathlib import Path
    model = hub.load_custom_model(Path("/path/to/checkpoint.pt"), config)

Core Classes
------------

ModelHub
~~~~~~~~

.. autoclass:: ModelHub
    :members:
    :show-inheritance:

    The main hub class that provides access to all model operations for a specific family.

    **Key Methods:**

    * :meth:`~ModelHub.get_archs` - List available architectures
    * :meth:`~ModelHub.get_arch_config` - Get architecture configuration
    * :meth:`~ModelHub.create_new_model` - Create newly initialized model
    * :meth:`~ModelHub.load_model` - Load model from asset card
    * :meth:`~ModelHub.load_custom_model` - Load model from custom checkpoint
    * :meth:`~ModelHub.iter_cards` - Iterate over available model cards

ModelHubAccessor
~~~~~~~~~~~~~~~~

.. autoclass:: ModelHubAccessor
    :members:
    :show-inheritance:

    Provides access to model hubs for specific families.
    Can be used by model implementors to create hub accessors for their model families, like :meth:`fairseq2.models.qwen.hub.get_qwen_model_hub`.

Global Functions
----------------

load_model
~~~~~~~~~~

.. autofunction:: load_model

    The main function for loading models across all families. Automatically determines
    the appropriate model family from the asset card.

    .. code-block:: python

        from fairseq2.models.hub import load_model

        # Load any model by name
        model = load_model("qwen25_7b")
        model = load_model("llama3_8b")
        model = load_model("mistral_7b")

Working with Model Families
---------------------------

Each model family provides its own hub accessor function:

Qwen Models
~~~~~~~~~~~

.. code-block:: python

    from fairseq2.models.qwen import get_qwen_model_hub

    hub = get_qwen_model_hub()

    # Available architectures
    archs = hub.get_archs()  # {'qwen25_0.5b', 'qwen25_1.5b', 'qwen25_3b', ...}

    # Get configuration for specific architecture
    config = hub.get_arch_config("qwen25_7b")

    # Create new model
    model = hub.create_new_model(config)

LLaMA Models
~~~~~~~~~~~~

.. code-block:: python

    from fairseq2.models.llama import get_llama_model_hub

    hub = get_llama_model_hub()

    # List available LLaMA architectures
    archs = hub.get_archs()

    # Load specific LLaMA model
    model = hub.load_model("llama3_8b")

Mistral Models
~~~~~~~~~~~~~~

.. code-block:: python

    from fairseq2.models.mistral import get_mistral_model_hub

    hub = get_mistral_model_hub()
    model = hub.load_model("mistral_7b")

Advanced Usage
--------------

Custom Model Loading
~~~~~~~~~~~~~~~~~~~~

Load models from custom checkpoints with specific configurations:

.. code-block:: python

    from pathlib import Path
    from fairseq2.models.qwen import get_qwen_model_hub

    hub = get_qwen_model_hub()

    # Get base configuration
    config = hub.get_arch_config("qwen25_7b")

    # Modify configuration if needed
    config.max_seq_len = 32768

    # Load from custom checkpoint
    checkpoint_path = Path("/path/to/my/checkpoint.pt")
    model = hub.load_custom_model(checkpoint_path, config)

Iterating Over Model Cards
~~~~~~~~~~~~~~~~~~~~~~~~~~

Discover all available models in a family:

.. code-block:: python

    from fairseq2.models.qwen import get_qwen_model_hub

    hub = get_qwen_model_hub()

    # List all Qwen model cards
    for card in hub.iter_cards():
        print(f"Model: {card.name}")
        print(f"  Architecture: {card.field('model_arch').as_(str)}")
        print(f"  Checkpoint: {card.field('checkpoint').as_(str)}")

Checkpoint Inspection
~~~~~~~~~~~~~~~~~~~~~

Iterate over checkpoint tensors without loading the full model:

.. code-block:: python

    from pathlib import Path
    from fairseq2.models.qwen import get_qwen_model_hub

    hub = get_qwen_model_hub()
    config = hub.get_arch_config("qwen25_7b")

    checkpoint_path = Path("/path/to/checkpoint.pt")

    # Inspect checkpoint contents
    for name, tensor in hub.iter_checkpoint(checkpoint_path, config):
        print(f"Parameter: {name}, Shape: {tensor.shape}")

Error Handling
--------------

Common Exceptions
~~~~~~~~~~~~~~~~~

.. autoexception:: ModelNotKnownError
    :show-inheritance:

    Raised when a requested model name is not found in the asset store.

.. autoexception:: ModelFamilyNotKnownError
    :show-inheritance:

    Raised when a model family is not registered or available.

.. autoexception:: ModelArchitectureNotKnownError
    :show-inheritance:

    Raised when a requested architecture is not available in the model family.

Example Error Handling
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from fairseq2.models.hub import load_model, ModelNotKnownError, ModelArchitectureNotKnownError
    from fairseq2.models.qwen import get_qwen_model_hub

    try:
        model = load_model("nonexistent_model")
    except ModelNotKnownError as e:
        print(f"Model not found: {e.name}")

    try:
        hub = get_qwen_model_hub()
        config = hub.get_arch_config("invalid_arch")
    except ModelArchitectureNotKnownError as e:
        print(f"Architecture not found: {e.arch}")
        print(f"Available architectures: {hub.get_archs()}")

See Also
--------

* :doc:`/guides/add_model` - Tutorial on adding new models
* :doc:`/basics/assets` - Understanding the asset system
* :doc:`/reference/fairseq2.models` - Models API overview
* :ref:`tokenizer_hub` for tokenizer hub reference documentation.
* :ref:`dataset_hub` for dataset hub reference documentation.
