.. _tokenizer_hub:

============================
fairseq2.data.tokenizers.hub
============================

.. currentmodule:: fairseq2.data.tokenizers.hub

The tokenizer hub provides a centralized way to manage and access tokenizers in fairseq2.
It offers functionality for discovering available tokenizers, loading tokenizers, and working
with custom tokenizer configurations.

Quick Reference
---------------

See :ref:`tokenizer` for detailed usage examples.

Core Classes
------------

TokenizerHub
~~~~~~~~~~~~

.. autoclass:: TokenizerHub
    :members:
    :show-inheritance:

    The main hub class for managing tokenizers. Provides methods for:

    - Listing available tokenizers (:meth:`iter_cards`)
    - Loading tokenizer configurations (:meth:`get_tokenizer_config`)
    - Loading tokenizers (:meth:`load_tokenizer`)
    - Loading custom tokenizers (:meth:`load_custom_tokenizer`)

    Example:

    .. code-block:: python

        from fairseq2.models.qwen import get_qwen_tokenizer_hub

        hub = get_qwen_tokenizer_hub()

        # list available tokenizers
        for card in hub.iter_cards():
            print(f"Found tokenizer: {card.name}")

        # directly load a tokenizer to ~/.cache/huggingface/models--qwen--qwen3-0.6b
        tokenizer = hub.load_tokenizer("qwen3_0.6b")

        # load a tokenizer configuration
        config = hub.get_tokenizer_config("qwen3_0.6b")
        
        # load a custom tokenizer from a path
        # hf download Qwen/Qwen3-0.6B --local-dir /data/pretrained_llms/qwen3_0.6b
        custom_path = Path("/data/pretrained_llms/qwen3_0.6b")
        custom_tokenizer = hub.load_custom_tokenizer(custom_path, config)

        # Generate some text
        text = "The future of AI is"
        encoder = custom_tokenizer.create_encoder()
        encoded = encoder(text)

        # Decode the text
        decoder = custom_tokenizer.create_decoder()
        decoded = decoder(encoded)

TokenizerHubAccessor
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TokenizerHubAccessor
    :members:
    :show-inheritance:

    Factory class for creating :class:`TokenizerHub` instances for specific tokenizer families.
    Can be used by tokenizer implementors to create hub accessors for their tokenizer families like :func:`fairseq2.models.qwen.get_qwen_tokenizer_hub`.

    Example:

    .. code-block:: python

        from fairseq2.data.tokenizers.hub import TokenizerHubAccessor
        from fairseq2.models.qwen import QwenTokenizer, QwenTokenizerConfig

        # the implementation of get_qwen_tokenizer_hub
        get_qwen_tokenizer_hub = TokenizerHubAccessor(
            "qwen",  # tokenizer family name
            QwenTokenizer,  # concrete tokenizer class
            QwenTokenizerConfig,  # concrete tokenizer config class
        )


Functions
---------

load_tokenizer
~~~~~~~~~~~~~~

.. autofunction:: load_tokenizer

    The global, family-agnostic function for loading tokenizers.
    This is a high-level function that handles all the complexities of tokenizer loading internally (via hub methods).

    Example:

    .. code-block:: python

        from fairseq2.data.tokenizers import load_tokenizer

        tokenizer = load_tokenizer("qwen3_0.6b")

Exceptions
----------

TokenizerNotKnownError
~~~~~~~~~~~~~~~~~~~~~~

.. autoexception:: TokenizerNotKnownError
    :show-inheritance:

    Raised when attempting to load a tokenizer that is not registered in the asset store.

TokenizerFamilyNotKnownError
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoexception:: TokenizerFamilyNotKnownError
    :show-inheritance:

    Raised when attempting to access a tokenizer family that is not registered in the system.

See Also
--------

- :ref:`dataset_hub` for dataset hub reference documentation.
- :ref:`api-models-hub` for model hub reference documentation.