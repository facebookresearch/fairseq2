fairseq2.data
=============
.. body

.. currentmodule:: fairseq2.data

``fairseq2.data`` provides a Python API to build a C++ :py:class:`DataPipeline`.

The dataloader will be able to leverage several threads,
working around Python GIL limitations,
and also providing better performance
than a pure Python dataloader.

Building a :py:class:`DataPipeline` looks like this::

    data = (
        text.read_text("file.tsv")
        .map(lambda x: str(x.split("\t")[1]).lower())
        .filter(lambda x: len(x) < 10)
    )

Functions to build a :py:class:`DataPipeline`:


.. autosummary::
    :toctree: generated/data

    DataPipeline
    DataPipelineBuilder

    list_files
    read_sequence
    read_zipped_records
    text.read_text
    FileMapper

    Collater
    CollateOptionsOverride

Public classes used in fairseq2 API:

.. autosummary::
    :toctree: generated/data

    CString
    PathLike
    StringLike
    ByteStreamError
    DataPipelineError
    RecordError
    VocabularyInfo

Helper methods:

.. autosummary::
    :toctree: generated/data

    get_last_failed_example
    is_string_like

fairseq2.data.text
~~~~~~~~~~~~~~~~~~

Tools to tokenize text, converting it from bytes to tensors.

.. currentmodule:: fairseq2.data.text

.. autosummary::
    :toctree: generated/data_text

    TextTokenizer
    MultilingualTextTokenizer
    TextTokenDecoder
    TextTokenEncoder

    StrSplitter
    StrToIntConverter
    StrToTensorConverter

    SentencePieceModel
    SentencePieceEncoder
    SentencePieceDecoder
    vocabulary_from_sentencepiece
    LineEnding
