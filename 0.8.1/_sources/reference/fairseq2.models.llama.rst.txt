.. _api-models-llama:

=====================
fairseq2.models.llama
=====================

.. currentmodule:: fairseq2.models.llama

The LLaMA module provides support for LLaMA language models from Meta AI.
It includes model configurations, hub access, tokenizers, and utilities for loading and working with LLaMA models.

Quick Start
-----------

.. code-block:: python

    from fairseq2.models.llama import get_llama_model_hub, get_llama_tokenizer_hub

    # Get the model hub
    hub = get_llama_model_hub()

    # Load a model
    model = hub.load_model("llama3_2_1b")

    # Load corresponding tokenizer (uses HuggingFace tokenizer by default)
    tokenizer = get_llama_tokenizer_hub().load_tokenizer("llama3_2_1b")

    # Generate some text
    text = "The future of AI is"
    encoder = tokenizer.create_encoder()
    encoded = encoder(text)
    # ... model inference code ...

Tokenizer
---------

LLaMA tokenizer in fairseq2 supports multiple implementations:

1. **HuggingFace Tokenizer (Default)**:
    The default and recommended implementation using HuggingFace's tokenizer.

    Asset Card Example:

    .. code-block:: yaml

        name: llama3
        tokenizer: "/path/to/Llama-3.1-8B"  # HuggingFace tokenizer directory
        tokenizer_family: llama

    Directory structure should contain e.g. ``config.json``, ``tokenizer.json``, ``tokenizer_config.json``, ``special_tokens_map.json``.

2. **Tiktoken Implementation**:
    Implementation using Tiktoken.

    Asset Card Example:

    .. code-block:: yaml

        name: tiktoken_llama_instruct
        tokenizer_config_override:
            impl: tiktoken
            use_eot: True  # For instruction models
        tokenizer_family: llama
        tokenizer: "/path/to/tokenizer.model"  # Tiktoken model file

3. **SentencePiece Implementation**:
    Implementation using SentencePiece (only available for LLaMA-1 and LLaMA-2).
    
    Asset Card Example:

    .. code-block:: yaml

        name: sp_llama
        tokenizer_config_override:
            impl: sp
        tokenizer_family: llama
        tokenizer: "/path/to/tokenizer.model"  # SentencePiece model file

Special Tokens
~~~~~~~~~~~~~~

The tokenizer handles several special tokens:

- ``<|begin_of_text|>`` - Beginning of text marker
- ``<|end_of_text|>`` - End of text marker (default)
- ``<|eot_id|>`` - End of turn marker (when ``use_eot=True``)
- ``<|start_header_id|>`` - Start of header
- ``<|end_header_id|>`` - End of header

For instruction models (e.g., ``llama3_2_1b_instruct``), ``use_eot=True`` is set by default, which means:

.. code-block:: python

    from fairseq2.data.tokenizers import load_tokenizer

    # Load instruct model tokenizer
    tokenizer = load_tokenizer("llama3_2_1b_instruct")
    
    # Will use <|eot_id|> as EOS token
    assert tokenizer._eos_token == "<|eot_id|>"

Tokenizer Modes
~~~~~~~~~~~~~~~

The tokenizer supports different modes via ``create_encoder(mode=...)``:

- ``default``: Adds BOS and EOS tokens
- ``prompt``: Adds BOS token only
- ``prompt_response``: Adds EOS token only
- ``as_is``: No special tokens added

.. code-block:: python

    encoder = tokenizer.create_encoder(mode="prompt")
    # Only adds <|begin_of_text|>

    encoder = tokenizer.create_encoder(mode="prompt_response")
    # Only adds <|eot_id|> or <|end_of_text|>

Model Configuration
-------------------

LLaMAConfig
~~~~~~~~~~~

.. autoclass:: LLaMAConfig
    :members:
    :show-inheritance:

Tokenizer Configuration
-----------------------

LLaMATokenizerConfig
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LLaMATokenizerConfig
    :members:
    :show-inheritance:

    Configuration for LLaMA tokenizer.

    **Key Parameters:**

    * ``impl`` - Implementation to use: "hg" (default), "tiktoken", or "sp"
    * ``use_eot`` - Whether to use ``<|eot_id|>`` as EOS token (True for instruction models)
    * ``split_regex`` - Custom regex pattern for tiktoken implementation

Complete Examples
-----------------

Using HuggingFace Tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from fairseq2.models.llama import get_llama_tokenizer_hub
    
    # Load default HuggingFace tokenizer
    tokenizer = get_llama_tokenizer_hub().load_tokenizer("llama3_2_1b")
    
    # Create encoder in different modes
    default_encoder = tokenizer.create_encoder()  # Adds BOS and EOS
    prompt_encoder = tokenizer.create_encoder(mode="prompt")  # Only BOS
    
    # Encode text
    text = "Hello, world!"
    tokens = default_encoder(text)

Using Tiktoken Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from fairseq2.models.llama import get_llama_tokenizer_hub
    from fairseq2.models.llama.tokenizer import LLaMATokenizerConfig
    from pathlib import Path

    # Configure tiktoken implementation
    config = LLaMATokenizerConfig(impl="tiktoken", use_eot=True)

    # Load tokenizer with custom config
    hub = get_llama_tokenizer_hub()
    tokenizer = hub.load_custom_tokenizer(Path("/path/to/tokenizer.model"), config)


Chat Template Support
~~~~~~~~~~~~~~~~~~~~~

The HuggingFace implementation includes support for chat templates through the HuggingFace tokenizer's ``apply_chat_template`` method:

.. code-block:: python

    from fairseq2.models.llama import get_llama_tokenizer_hub

    # Load tokenizer
    tokenizer = get_llama_tokenizer_hub().load_tokenizer("llama3_2_1b")

    # Prepare chat messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
        {"role": "assistant", "content": "Why did the chicken cross the road?"}
    ]

    # Format using chat template
    formatted_text = tokenizer._model._tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Then encode the formatted text
    encoder = tokenizer.create_encoder()
    tokens = encoder(formatted_text)
    
See Also
--------

* :doc:`/reference/fairseq2.models.hub` - Model hub API reference
* :doc:`/guides/add_model` - Tutorial on adding new models
* :doc:`/basics/assets` - Understanding the asset system
