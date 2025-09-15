.. _tokenizer:

fairseq2.data.tokenizers
========================

.. currentmodule:: fairseq2.data.tokenizers

The tokenizer has multiple concrete implementations for different tokenization algorithms.
The main :class:`Tokenizer` interface defines the contract for creating encoders and decoders, while concrete implementations
handle specific tokenization methods like SentencePiece and tiktoken.

.. mermaid::

   classDiagram
       class Tokenizer {
           <<abstract>>
           +create_encoder(task, lang, mode, device)*
           +create_raw_encoder(device)*
           +create_decoder(skip_special_tokens)*
           +vocab_info: VocabularyInfo*
       }

       class BasicSentencePieceTokenizer {
           -_model: SentencePieceModel
           -_vocab_info: VocabularyInfo
           +create_encoder(task, lang, mode, device)
           +create_raw_encoder(device)
           +create_decoder(skip_special_tokens)
       }

       class RawSentencePieceTokenizer {
           -_model: SentencePieceModel
           -_vocab_info: VocabularyInfo
           +create_encoder(task, lang, mode, device)
           +create_raw_encoder(device)
           +create_decoder(skip_special_tokens)
       }

       class TiktokenTokenizer {
           -_model: TiktokenModel
           -_vocab_info: VocabularyInfo
           +create_encoder(task, lang, mode, device)
           +create_raw_encoder(device)
           +create_decoder(skip_special_tokens)
       }

       class CharTokenizer {
           -_vocab_info: VocabularyInfo
           +create_encoder(task, lang, mode, device)
           +create_raw_encoder(device)
           +create_decoder(skip_special_tokens)
       }

       Tokenizer <|-- BasicSentencePieceTokenizer
       Tokenizer <|-- RawSentencePieceTokenizer
       Tokenizer <|-- TiktokenTokenizer
       Tokenizer <|-- CharTokenizer



Quick Start
-----------

Loading a Tokenizer
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from fairseq2.data.tokenizers import load_tokenizer

    tokenizer = load_tokenizer("qwen3_0.6b")


Listing Available Tokenizers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To list available tokenizers of a specific family (e.g., Qwen):

.. code-block:: python

    from fairseq2.models.qwen import get_qwen_tokenizer_hub

    hub = get_qwen_tokenizer_hub()

    for card in hub.iter_cards():
        print(f"Found tokenizer: {card.name}")


Loading a Specific Model's Tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from fairseq2.models.qwen import get_qwen_tokenizer_hub

    hub = get_qwen_tokenizer_hub()

    # directly load a tokenizer to ~/.cache/huggingface/models--qwen--qwen3-0.6b
    tokenizer = hub.load_tokenizer("qwen3_0.6b")


This loads the tokenizer and its associated vocabulary for the specified model.


Using TokenizerHub
~~~~~~~~~~~~~~~~~~

:class:`TokenizerHub` provides more advanced/customized operations for working with tokenizers.
This is helpful if you want to implement your own tokenizer, and configuration.
Here's how to use it with Qwen tokenizers (you can adapt this for your own tokenizer family):

.. code-block:: python

    from fairseq2.data.tokenizers.hub import TokenizerHubAccessor
    from fairseq2.models.qwen import QwenTokenizer, QwenTokenizerConfig
    from pathlib import Path

    # when implementing your own tokenizer family, you can create a similar helper function
    # to load the hub for that family.
    # behind the scene, get_qwen_tokenizer_hub is implemented like this:
    get_qwen_tokenizer_hub = TokenizerHubAccessor(
        "qwen",  # tokenizer family name
        QwenTokenizer,  # concrete tokenizer class
        QwenTokenizerConfig,  # concrete tokenizer config class
    )
    hub = get_qwen_tokenizer_hub()

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


Listing Available Tokenizers (CLI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # List tokenizers from command line
    python -m fairseq2.assets list --kind tokenizer

.. toctree::
    :maxdepth: 1

    hub