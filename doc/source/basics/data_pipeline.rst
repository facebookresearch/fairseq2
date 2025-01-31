.. _basics-data-pipeline:

:octicon:`database` Data Pipeline
=================================

Data pipelines in fairseq2 provide an efficient way to process and transform data for machine learning tasks.
The implementation leverages multiple threads to work around Python's Global Interpreter Lock (GIL) limitations,
resulting in better performance than pure Python dataloaders.

Basic Pipeline Structure
^^^^^^^^^^^^^^^^^^^^^^^^

A data pipeline consists of a series of operations that transform data. Here's a basic example::

    data = (
        text.read_text("file.tsv")
        .map(lambda x: str(x.split("\t")[1]).lower())
        .filter(lambda x: len(x) < 10)
    )

.. mermaid::

   graph LR
      A[read_text] --> B[map]
      B --> C[filter]
      style A fill:#f9f,stroke:#333
      style B fill:#bbf,stroke:#333
      style C fill:#bfb,stroke:#333

.. dropdown:: A more complex pipeline that can be built w/ fairseq2 as a diagram
   :icon: package
   :animate: fade-in

   .. image:: /_static/img/data/complex_data_pipeline_example.svg
      :alt: A more complex pipeline that can be built w/ fairseq2

.. _basics/data-pipeline/column-selection:

Column Selection
^^^^^^^^^^^^^^^^

Data items in the pipeline can be tuples or Python dictionaries. Many operators support a `selector` argument to specify which column to process:

- For tuples: ``"[3]"`` selects the fourth element (0-based indexing)
- For dictionaries: ``"foo"`` selects the value for key ``"foo"``
- Nested selectors: Use ``.`` to traverse nested structures (e.g., ``"foo[1].y"``)

Example with nested data::

    data = {"foo": [{"x": 1, "y": 2}, {"x": 3, "y": 4, "z": 5}], "bar": 6}
    # "foo[1].y" selects 4
    # "bar" selects 6

.. mermaid::

   graph TD
      A[Input Dictionary] --> B[foo]
      A --> C[bar: 6]
      B --> D[List Index 0]
      B --> E[List Index 1]
      D --> F[x: 1]
      D --> G[y: 2]
      E --> H[x: 3]
      E --> I[y: 4]
      E --> J[z: 5]
      style I fill:#f96,stroke:#333
      style C fill:#f96,stroke:#333

.. _basics/data-pipeline/pipeline-types:

Pipeline Types
^^^^^^^^^^^^^^

fairseq2 supports three types of pipelines:

1. **Finite Pipelines**: Standard pipelines that terminate after processing all data
2. **Pseudo-infinite Pipelines**: Created using ``DataPipeline.count`` or ``DataPipeline.constant``
3. **Infinite Pipelines**: Created using ``DataPipelineBuilder.repeat`` without arguments

.. mermaid::

   graph TD
      subgraph Finite
         A[read_sequence] --> B[End]
      end
      subgraph Pseudo-infinite
         C[constant/count] --> D[Stops with other pipelines]
      end
      subgraph Infinite
         E[repeat] --> F[Never ends]
      end

.. _basics/data-pipeline/combining-pipelines:

Combining Pipelines
^^^^^^^^^^^^^^^^^^^

fairseq2 provides several ways to combine pipelines:

1. **Round Robin**: Alternates between pipelines::

    pipeline1 = DataPipeline.constant(0).and_return()
    pipeline2 = read_sequence([1, 2, 3]).and_return()
    
    for example in DataPipeline.round_robin(
        [pipeline1, pipeline2],
        stop_at_shortest=False,  # Continue until longest pipeline ends
        allow_repeats=True       # Allow repeating from finished pipelines
    ).and_return():
        print(example)

    # round_robin yields: 0, 1, 0, 2, 0, 3

2. **Zip**: Combines examples from multiple pipelines::

    pipeline1 = read_sequence([0]).repeat().and_return()
    pipeline2 = read_sequence([1, 2, 3]).and_return()
    
    for example in DataPipeline.zip(
        [pipeline1, pipeline2],
        names=["a", "b"],           # Name the columns
        zip_to_shortest=False,      # Continue until longest pipeline ends
        flatten=False,              # Keep structure as is
        disable_parallelism=False   # Allow parallel processing
    ).and_return():
        print(example)

    # Yields: {"a": 0, "b": 1}, {"a": 0, "b": 2}, {"a": 0, "b": 3}

3. **Sample**: Randomly samples from pipelines based on weights::

    pipeline1 = read_sequence([0]).repeat().and_return()
    pipeline2 = read_sequence([1, 2, 3]).and_return()
    
    for example in DataPipeline.sample([pipeline1, pipeline2], weights=[0.5, 0.5]).and_return():
        print(example)

.. mermaid::

   graph TD
      subgraph Round Robin
         A1[Pipeline 1] --> C1{Alternate}
         B1[Pipeline 2] --> C1
         C1 --> D1[Output]
      end
      subgraph Zip
         A2[Pipeline 1] --> C2((Combine))
         B2[Pipeline 2] --> C2
         C2 --> D2[Output]
      end
      subgraph Sample
         A3[Pipeline 1] --> C3{Random Select}
         B3[Pipeline 2] --> C3
         C3 --> D3[Output]
      end

More Features
^^^^^^^^^^^^^

Shuffling
~~~~~~~~~

fairseq2 provides flexible shuffling capabilities through the ``shuffle`` operator:

.. code-block:: python

    # Basic shuffling with a window size
    pipeline = (
        read_sequence(data)
        .shuffle(shuffle_window=1000)  # Shuffle using a 1000-example buffer
        .and_return()
    )

    # Shuffle between epochs
    for epoch in range(3):
        pipeline.reset()  # By default, this re-shuffles data
        for item in pipeline:
            process(item)

    # Disable shuffling between epochs
    pipeline.reset(reset_rng=True)  # Keep the same order

The shuffle operator maintains a buffer of the specified size.
When requesting the next example, it randomly samples from this buffer and replaces the selected example with a new one from the source.
Setting ``shuffle_window=0`` loads all examples into memory for full shuffling.

Bucketing
~~~~~~~~~

Bucketing helps handle variable-length sequences efficiently. There are several bucketing strategies:

1. **Fixed-size Bucketing**: Combine a fixed number of examples

.. code-block:: python

    pipeline = (
        read_sequence(data)
        .bucket(bucket_size=32, drop_remainder=True)  # Combine 32 examples into one bucket
        .and_return()
    )

2. **Length-based Bucketing**: Group sequences of similar lengths

.. code-block:: python

    from fairseq2.data import create_bucket_sizes

    # Create optimal bucket sizes
    bucket_sizes = create_bucket_sizes(
        max_num_elements=1024,   # Max elements per bucket
        max_seq_len=128,         # Max sequence length
        min_seq_len=1,           # Min sequence length
        num_seqs_multiple_of=8   # Ensure bucket sizes are multiples of 8
    )

    # Use bucketing in pipeline
    pipeline = (
        read_sequence(data)
        .bucket_by_length(
            bucket_sizes,
            selector="length",             # Column containing sequence lengths
            skip_above_max_examples=True,  # Skip sequences longer than max_seq_len
            drop_remainder=False           # Keep partial buckets
        )
        .and_return()
    )

3. **Dynamic Bucketing**: Combine examples based on a cost function

.. code-block:: python

    def sequence_cost(example):
        return len(example["text"])

    pipeline = (
        read_sequence(data)
        .dynamic_bucket(
            threshold=1024,        # Target bucket size
            cost_fn=sequence_cost, # Function to compute example cost
            min_num_examples=16,   # Min examples per bucket
            max_num_examples=64,   # Max examples per bucket
            drop_remainder=False   # Keep partial buckets
        )
        .and_return()
    )


This approach efficiently handles variable-length sequences while maintaining appropriate batch sizes for training.

There are more features in fairseq2's data pipeline:

- **Prefetching**: Load data ahead of time for better performance
- **State Management**: Save and restore pipeline state for resumable processing

.. note::
   When combining pseudo-infinite pipelines with finite ones, the pseudo-infinite pipeline will stop when the finite pipeline ends.
   For truly infinite behavior, use ``repeat()`` without arguments.

For more technical details, see :doc:`/reference/api/fairseq2.data/index`.

See Also
--------

- Jupyter Notebook Example: :doc:`/notebooks/datapipeline`

Error Handling
~~~~~~~~~~~~~~

fairseq2 provides robust error handling capabilities:

.. code-block:: python

    # Control maximum number of warnings before raising an error
    pipeline = (
        read_sequence(data)
        .filter(error_prone_function)
        .and_return(max_num_warnings=3)  # Allow up to 3 errors before failing
    )

    # Access the last failed example for debugging
    from fairseq2.data import get_last_failed_example

    try:
        for item in pipeline:
            process(item)
    except Exception as e:
        failed_example = get_last_failed_example()
        print(f"Failed on example: {failed_example}")

State Management
~~~~~~~~~~~~~~~~

The pipeline state can be saved and restored for resumable processing:

.. code-block:: python

    pipeline = read_sequence(data).and_return()
    
    # Save state after processing some examples
    state_dict = pipeline.state_dict()
    
    # Restore state later
    pipeline.load_state_dict(state_dict)
