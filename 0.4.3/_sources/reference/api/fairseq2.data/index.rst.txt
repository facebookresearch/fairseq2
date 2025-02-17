=============
fairseq2.data
=============

.. module:: fairseq2.data

The data pipeline module provides the core data processing functionality in fairseq2.

.. autoclasstree:: fairseq2.data
   :full:
   :zoom:

.. toctree::
    :maxdepth: 1

    text/index

Classes
-------

.. autoclass:: DataPipeline
   :members:
   :special-members: __iter__

.. autoclass:: DataPipelineBuilder
   :members:

.. autoclass:: Collater
   :members:
   :special-members: __call__

.. autoclass:: CollateOptionsOverride
   :members:
   :special-members: __init__

.. autoclass:: FileMapper
   :members:
   :special-members: __init__, __call__

.. autoclass:: SequenceData
   :members:

.. autoclass:: FileMapperOutput
   :members:

Functions
---------

.. autofunction:: create_bucket_sizes

.. autofunction:: get_last_failed_example

.. autofunction:: list_files

.. autofunction:: read_sequence

.. autofunction:: read_zipped_records

.. autofunction:: read_iterator

Exceptions
----------

.. autoclass:: DataPipelineError
   :members:

.. autoclass:: ByteStreamError
   :members:

.. autoclass:: RecordError
   :members:

Examples
--------

Creating a Basic Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from fairseq2.data import read_sequence, DataPipeline

    # Create a simple pipeline that processes numbers
    pipeline = (
        read_sequence([1, 2, 3, 4, 5])
        .map(lambda x: x * 2)
        .filter(lambda x: x > 5)
        .and_return()
    )

    # Iterate over the results
    for item in pipeline:
        print(item)  # Outputs: 6, 8, 10

Using Column Selection
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Process structured data with column selection
    data = [
        {"text": "Hello", "label": 1},
        {"text": "World", "label": 0}
    ]

    pipeline = (
        read_sequence(data)
        .map(lambda x: x.upper(), selector="text")
        .and_return()
    )

    # Results will have uppercase text but unchanged labels
    # [{"text": "HELLO", "label": 1}, {"text": "WORLD", "label": 0}]

Combining Pipelines
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create two pipelines
    p1 = read_sequence([1, 2, 3]).and_return()
    p2 = read_sequence(['a', 'b', 'c']).and_return()

    # Zip them together with names
    combined = DataPipeline.zip(
        [p1, p2],
        names=["numbers", "letters"]
    ).and_return()

    # Results: [
    #   {"numbers": 1, "letters": 'a'},
    #   {"numbers": 2, "letters": 'b'},
    #   {"numbers": 3, "letters": 'c'}
    # ]

Using Bucketing
^^^^^^^^^^^^^^^

.. code-block:: python

    from fairseq2.data import create_bucket_sizes

    # Create optimal bucket sizes for sequence processing
    bucket_sizes = create_bucket_sizes(
        max_num_elements=1024,
        max_seq_len=128,
        min_seq_len=1,
        num_seqs_multiple_of=8
    )

    # Use bucketing in a pipeline
    pipeline = (
        read_sequence(data)
        .bucket_by_length(
            bucket_sizes,
            selector="text",
            drop_remainder=False
        )
        .and_return()
    )

State Management
^^^^^^^^^^^^^^^^

.. code-block:: python

    # Save pipeline state
    state = pipeline.state_dict()

    # Load pipeline state
    new_pipeline = create_pipeline()  # Create a new pipeline
    new_pipeline.load_state_dict(state)  # Restore the state

See Also
--------

* :ref:`basics-data-pipeline` - Basic introduction to data pipeline
