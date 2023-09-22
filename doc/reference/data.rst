fairseq2.data
=============
.. body

.. currentmodule:: fairseq2.data

.. autosummary::
    :toctree: generated/data

    DataPipeline
    DataPipelineBuilder
    CString

    list_files
    read_sequence
    read_zipped_records

    Collater
    FileMapper

    VocabularyInfo
    ByteStreamError
    CollateOptionsOverride
    DataPipelineError
    RecordError
    PathLike
    StringLike
    get_last_failed_example
    is_string_like

fairseq2.data.text
~~~~~~~~~~~~~~~~~~

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
    read_text
