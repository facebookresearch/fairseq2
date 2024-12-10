fairseq2.data
=============
.. body

.. currentmodule:: fairseq2.data

``fairseq2.data`` provides a Python API to build a C++ :py:class:`DataPipeline`.

The dataloader will be able to leverage several threads,
working around Python Global Interpreter Lock limitations,
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

Column syntax
~~~~~~~~~~~~~

The data items going through the pipeline don't have to be flat tensors, but can be tuples, or python dictionaries.
Several operators have a syntax to specify a specific column of the input data.
Notably the :py:func:`DataPipelineBuilder.map` operator
has a `selector` argument to choose the column to apply the function to.

If the data item is a tuple,
then the selector ``"[3]"`` selects the third column.
If the data item is a dictionary, then ``"foo"`` will select the value corresponding to the key ``"foo"``.
You can nest selectors using ``.`` to separate key selectors, following a python-like syntax.
For a data item ``{"foo": [{"x": 1, "y": 2}, {"x": 3, "y": 4, "z": 5}], "bar": 6}``,
the selector ``"foo[1].y"`` referes to  the value 4.

Functions that accepts several selectors,
accept them as a comma separated list of selectors.
For example ``.map(lambda x: x * 10, selector="foo[1].y,bar")``
will multiply the values 4 and 6 by 10, but leave others unmodified.

Pseudo-infinite and Infinite Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :py:func:`DataPipeline.count` and :py:func:`DataPipeline.constant` static methods create pseudo-infinite pipelines.
When used with operators that combine multiple pipelines (e.g. :py:func:`DataPipeline.sample`,
:py:func:`DataPipeline.round_robin`, :py:func:`DataPipeline.zip`),
they will only yield examples as long as the other pipelines yield examples.

For example::

    from fairseq2.data import DataPipeline, read_sequence

    pipeline1 = DataPipeline.constant(0).and_return()
    pipeline2 = read_sequence([1, 2, 3]).and_return()

    for example in DataPipeline.round_robin(pipeline1, pipeline2).and_return():
        print(example)

only produces 0, 1, 0, 2, 0, 3.

Infinite pipelines (pipelines created through :py:func:`DataPipelineBuilder.repeat` with no arguments)
do not exhibit this behavior; they will yield examples indefinitely even when combined with other pipelines.

For example::

    from fairseq2.data import DataPipeline, read_sequence

    pipeline1 = read_sequence([0]).repeat().and_return()
    pipeline2 = read_sequence([1, 2, 3]).and_return()

    for example in DataPipeline.round_robin(pipeline1, pipeline2).and_return():
        print(example)

produces 0, 1, 0, 2, 0, 3, 0, 1, 0, 2, 0, 3... indefinitely.


Public classes used in fairseq2 API:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/data

    ByteStreamError
    DataPipelineError
    RecordError
    VocabularyInfo

Helper methods:

.. autosummary::
    :toctree: generated/data

    get_last_failed_example

fairseq2.data.text
~~~~~~~~~~~~~~~~~~

Tools to tokenize text, converting it from bytes to tensors.

.. currentmodule:: fairseq2.data.text

.. autosummary::
    :toctree: generated/data_text

    TextTokenizer
    TextTokenDecoder
    TextTokenEncoder

    StrSplitter
    StrToIntConverter
    StrToTensorConverter

    SentencePieceModel
    SentencePieceEncoder
    SentencePieceDecoder
    vocab_info_from_sentencepiece
    LineEnding
