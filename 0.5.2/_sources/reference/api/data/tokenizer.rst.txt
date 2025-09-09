.. _tokenizer:

Tokenizers
==========

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

    tokenizer = load_tokenizer("qwen3_1.7b")


Listing Available Tokenizers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # List tokenizers from command line
    python -m fairseq2.assets list --kind tokenizer
