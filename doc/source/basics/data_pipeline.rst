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
    
    for example in DataPipeline.round_robin(pipeline1, pipeline2).and_return():
        print(example)

    # round_robin yields: 0, 1, 0, 2, 0, 3

2. **Zip**: Combines examples from multiple pipelines::

    pipeline1 = read_sequence([0]).repeat().and_return()
    pipeline2 = read_sequence([1, 2, 3]).and_return()
    
    for example in DataPipeline.zip(pipeline1, pipeline2, names=["a", "b"]).and_return():
        print(example)

    # Yields: {"a": 0, "b": 1}, {"a": 0, "b": 2}, {"a": 0, "b": 3}

3. **Sample**: Randomly samples from pipelines based on weights::

    pipeline1 = read_sequence([0]).repeat().and_return()
    pipeline2 = read_sequence([1, 2, 3]).and_return()
    
    for example in DataPipeline.sample(pipeline1, pipeline2, weights=[0.5, 0.5]).and_return():
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

Advanced Features
^^^^^^^^^^^^^^^^^

- **Bucketing**: Group examples by size or custom criteria
- **Shuffling**: Randomize data order with configurable window size
- **Prefetching**: Load data ahead of time for better performance
- **State Management**: Save and restore pipeline state for resumable processing

.. note::
   When combining pseudo-infinite pipelines with finite ones, the pseudo-infinite pipeline will stop when the finite pipeline ends.
   For truly infinite behavior, use ``repeat()`` without arguments.

For more technical details, see :doc:`/reference/api/fairseq2.data/data_pipeline`.