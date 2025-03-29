.. _basics-parquet-dataloader:

:octicon:`database` Parquet Data Loading
========================================

`Parquet <https://parquet.apache.org/docs/>`_ is a popular binary columnar file format optimized for data storage and large-scale distributed data processing.

.. note::

    Consider using the Parquet format when dealing with complex or nested data structures, such as embeddings, tokens, audio files, bytes, etc., as it enables their efficient and self-contained representation.

A Parquet dataset is a collection of Parquet files that can be partitioned or not. Each Parquet file consists of row groups. Roughly speaking, a row group is the smallest piece of a Parquet file that can be read into memory. Since Parquet is a columnar format, we can flexibly and efficiently read only a subset of columns from a row group.

.. note::

    The row group size, when writing a Parquet file, should be chosen carefully to balance the trade-off between memory usage, read performance, and shuffling quality. As a rule of thumb, a good initial recommendation is to adjust the row group size so that each row group is between 50MB and 500MB.

This module provides an efficient and flexible data loading pipeline for Apache Parquet datasets in fairseq2.
The present tooling is general purpose and can be combined with various downstream workflows for large-scale machine learning workloads with features like sharding, filtering, column selection, and dynamic batching.

**Requirements**: Install the Arrow dependencies with ``pip install fairseq2[arrow]``, since we rely on the
`pyarrow <https://arrow.apache.org/docs/python/index.html>`_ library to interface with parquet files.

Architecture Overview
---------------------

The data loading pipeline is organized into three main components:

1. **Fragment Streaming**: Produces a stream of dataset fragments (files or row groups)
2. **Fragment Loading**: Reads data from these fragments into PyArrow tables
3. **Table Bucketing**: Combines and processes tables into batches

Each component has its own configuration class to control behavior.

.. mermaid::

    graph LR
        A[Fragment Streaming] --> B[Fragment Loading]
        B --> C[Table Bucketing]

Fragment Streaming
------------------

The ``FragmentStreamingConfig`` class defines how fragments are streamed from a Parquet dataset.

Basic Usage
^^^^^^^^^^^

.. code-block:: python

    from fairseq2.data.parquet.fragment_streaming import (
        FragmentStreamingConfig, ParquetFragmentStreamer
    )

    # Simple configuration
    config = FragmentStreamingConfig(
        parquet_path="/path/to/dataset/",
        nb_epochs=2,
        split_to_row_groups=True,  # Work with row groups instead of files. Set to False makes a fragment correspond to a file.
        fragment_shuffle_window=100,  # Shuffle within a window of 100 fragments
        seed=42,  # For reproducibility
    )

    # Create the streamer
    streamer = ParquetFragmentStreamer(config=config)

    # Build a pipeline for a specific rank/world_size (for distributed training)
    fragment_pipeline = streamer.build_pipeline(rank=0, world_size=1).and_return()

Shuffling Options
^^^^^^^^^^^^^^^^^

The module provides several shuffling strategies:

.. code-block:: python

    # No shuffling
    config = FragmentStreamingConfig(..., fragment_shuffle_window=0)

    # Global shuffling (requires loading all metadata up front)
    config = FragmentStreamingConfig(..., fragment_shuffle_window=-1)

    # Local shuffling within a window (faster start time)
    config = FragmentStreamingConfig(..., fragment_shuffle_window=100)

    # Circular shift for distributed training
    config = FragmentStreamingConfig(
        ...,
        files_circular_shift=True,  # Different ranks start at different positions
        fragment_shuffle_window=100
    )

.. note::

    How shuffling works:
    
    - For non-zero positive ``fragment_shuffle_window`` value, all dataset files will be shuffled globally (and this shuffling will be different from one epoch to another).
    - Next, each file will be split into row groups and shuffled locally within ``fragment_shuffle_window``.
    
    Note that the global shuffling needs all parquet files' metadata upfront, which can be expensive for remote large datasets.
    However, if ``fragment_shuffle_window`` is set to a small value (e.g. ~ average number of fragments per file * 5), the time to the first batch will be shorter.
    The metadata fetching will be done on the fly in that case.
    
    Also note that the shuffling behavior is seeded to be completely deterministic by the ``seed`` parameter.
    Thus if one resets a pipeline with the same ``seed`` value, the exactly same shuffling will be applied.

Sharding
^^^^^^^^

We can shard a dataset at fragment level using the ``rank`` and ``world_size`` parameters in ``build_pipeline``:

.. code-block:: python

    # Create a pipeline for a specific rank in distributed training
    pipeline = streamer.build_pipeline(rank=rank, world_size=world_size)

This sharding will typically be uneven -- different ranks may receive different numbers of rows.
Therefore we recommend using ``nb_epochs=None`` for infinite loops in large training runs.
Alternatively, if parquet dataloading is not the bottleneck, you can stream all fragments without sharding -- load them in memory and only then shard them at the row level to get more uniform sharding.

Filtering Datasets
^^^^^^^^^^^^^^^^^^

You can filter the dataset using PyArrow expressions:

.. code-block:: python

    import pyarrow.compute as pc

    # Filter by partition
    config = FragmentStreamingConfig(
        partition_filters='pc.is_in(pc.field("split"), pa.array(["dev", "test"]))'
    )

    # Multiple filters
    config = FragmentStreamingConfig(
        partition_filters=[
            'pc.field("split") == "train"',
            'pc.field("language") == "en"'
        ]
    )

    # Complex filters
    config = FragmentStreamingConfig(
        partition_filters='pc.greater(pc.field("date"), pc.scalar("2023-01-01"))'
    )

.. note::

    Make sure that the filters are applied to the partition columns.
    If you want to apply filters to non-partition columns, you will need to apply the filters during the loading process.

Fragment Loading
----------------

The ``FragmentLoadingConfig`` defines how data is loaded from fragments.

Column Selection
^^^^^^^^^^^^^^^^

Use the ``NamedColumns`` class to define which columns to load and how to rename them:

.. code-block:: python

    from dataclasses import dataclass, field
    from typing import List
    from fairseq2.data.parquet.fragment_loading import (
        FragmentLoadingConfig, NamedColumns, ParquetFragmentLoader
    )

    @dataclass
    class MyColumns(NamedColumns):
        # Format: new_name: original_column_name
        text: str = "source_text"
        label: str = "target_label"
        extra_columns: List[str] = field(default_factory=lambda: ["metadata", "timestamp"])

    # Create the loading config
    loading_config = FragmentLoadingConfig(
        columns=MyColumns(),
        add_fragment_traces=True,  # Add tracking columns
        drop_null=True,  # Drop rows with null values
        nb_prefetch=2,  # Prefetch 2 fragments
        num_parallel_fragments=4,  # Process 4 fragments in parallel
    )

    # Build the loading pipeline
    loader = ParquetFragmentLoader(config=loading_config).build_pipeline(fragment_pipeline)

Filtering Loaded Data
^^^^^^^^^^^^^^^^^^^^^

You can filter data after loading:

.. code-block:: python

    loading_config = FragmentLoadingConfig(
        columns=MyColumns(),
        filters='pc.greater(pc.list_value_length(pc.field("text")), 10)'  # Rows with text length > 10
    )

.. note::

    Note that this is another layer of filtering different from the ``partition_filters`` for fragment streaming.

Table Bucketing
---------------

The ``TableBucketingConfig`` controls how tables are combined and batched:

.. code-block:: python

    from fairseq2.data.parquet.table_bucketing import (
        TableBucketingConfig, TableBucketer
    )

    # Create bucketing config
    bucketing_config = TableBucketingConfig(
        target_table_size=1000,  # Aim for tables with 1000 rows
        min_fragment_number=2,   # Combine at least 2 fragments
        max_fragment_number=10,  # Combine at most 10 fragments
        shuffle=True,            # Shuffle rows in memory
        batch_size=32            # Return batches of 32 rows
    )

    # Apply bucketing
    bucketer = TableBucketer(bucketing_config)
    final_pipeline = bucketer.apply(loading_pipeline)

    # Iterate through batches
    for batch in final_pipeline.and_return():
        # batch is a PyArrow Table
        print(batch.column_names)
        print(len(batch))

Complete Pipeline Examples
--------------------------

Basic End-to-End Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from fairseq2.data.parquet import (
        BasicDataLoadingConfig,
        build_basic_parquet_data_pipeline,
        FragmentStreamingConfig,
        FragmentLoadingConfig,
        TableBucketingConfig
    )

    # Configure the entire pipeline
    config = BasicDataLoadingConfig(
        fragment_stream_config=FragmentStreamingConfig(
            parquet_path="/path/to/dataset/",
            partition_filters='pc.field("split") == "train"',
            nb_epochs=None,  # Infinite iterations
            fragment_shuffle_window=100
        ),
        fragment_load_config=FragmentLoadingConfig(
            columns=MyColumns(),
            nb_prefetch=2,
            num_parallel_fragments=3
        ),
        table_bucketing_config=TableBucketingConfig(
            target_table_size=1000,
            min_fragment_number=2,
            max_fragment_number=10,
            shuffle=True,
            batch_size=32
        ),
    )

    # Create the pipeline
    pipeline = build_basic_parquet_data_pipeline(config).and_return()

    # Use the pipeline
    for batch in pipeline:
        # Process the batch
        pass

Distributed Training Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch.distributed as dist

    # Get distributed info
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    config = BasicDataLoadingConfig(
        fragment_stream_config=FragmentStreamingConfig(
            parquet_path="/path/to/dataset/",
            fragment_shuffle_window=100,
            files_circular_shift=True  # Different ranks start at different positions
        ),
        # ... other configs
    )

    # Create a pipeline for this rank
    pipeline = build_basic_parquet_data_pipeline(
        config, rank=rank, world_size=world_size
    ).and_return()

Working with PyArrow Tables
---------------------------

PyArrow tables can be converted to various formats:

.. code-block:: python

    # Convert to pandas
    df = batch.to_pandas()

    # Convert to dictionary
    batch_dict = batch.to_pydict()

    # Convert to torch tensors
    from fairseq2.data.parquet.utils import pyarrow_table_to_torch_dict
    tensor_dict = pyarrow_table_to_torch_dict(batch)

    # Using Polars (fast with zero-copy)
    import polars as pl
    polars_df = pl.from_arrow(batch, rechunk=False)

    # Convert to list of dictionaries (rows)
    rows = batch.to_pylist()
    # Or using polars (usually much faster)
    rows = pl.from_arrow(batch, rechunk=False).to_dicts()

.. note::

    - Using `polars <https://docs.pola.rs/>`_, one can use ``pl.from_arrow(pa_table, rechunk=False)`` to convert into a polars dataframe (with almost zero memory copy)
    - ``pa.Table.to_pylist()`` or ``pl.from_arrow(...).to_dicts()`` (usually much faster) to convert into a list of dictionaries
    - ``parquet/utils.py:pyarrow_table_to_torch_dict`` to convert pyarrow table into a dictionary of cpu torch tensors (best effort)

Performance Considerations
--------------------------

Optimizing for Large Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For large, remote datasets:

.. code-block:: python

    config = FragmentStreamingConfig(
        # Avoid global shuffling which requires loading all metadata
        fragment_shuffle_window=200,  # Use local shuffling
        split_to_row_groups=True,     # Work with smaller row groups
    )

    loading_config = FragmentLoadingConfig(
        # Only load needed columns
        columns=MyColumns(text="source", label="target"),
        # Cache data to reduce memory usage with large remote datasets
        cache=True,
        # Parallelize fragment loading
        num_parallel_fragments=4,
        # Prefetch to hide I/O latency
        nb_prefetch=2
    )

Memory Management
^^^^^^^^^^^^^^^^^

For memory-intensive workloads:

.. code-block:: python

    loading_config = FragmentLoadingConfig(
        # Enable caching to disk for large tables
        cache=True,
        # Parallelize and prefetch for efficiency
        num_parallel_fragments=2,
        # Column pruning to reduce memory footprint
        columns=MyColumns(text="source")  # Only load needed columns
    )

    bucketing_config = TableBucketingConfig(
        # Control memory usage directly (in MB)
        target_table_memory=250,  # Limit each table to 250MB
        # Set boundaries for fragment combining
        min_fragment_number=1,
        max_fragment_number=5,
        # Apply smaller batches for processing
        batch_size=16,
        # Enable memory mapping for large tables
        cache=True,
        # Consider setting combine_chunks=False for very large datasets
        combine_chunks=True
    )

The ``target_table_memory`` parameter provides direct control over the memory footprint:

- Specified in megabytes (MB)
- Controls how many fragments are loaded and concatenated before processing
- Adapts to data complexity (variable-length text, lists, etc.)
- More predictable memory peaks than row-based approaches
- Better handles cases where row count doesn't correlate linearly with memory usage

As alternatives, you can also use:

- ``target_table_size``: Controls the number of rows (instead of memory)
- ``target_total_length``: Controls the total token length across all columns

For maximum memory efficiency, combine with:

- Memory mapping: ``cache=True`` to store tables on disk
- Column pruning: Only load needed columns
- Chunk management: Consider ``combine_chunks=False`` for very large datasets

Transformations
---------------

You can apply custom transformations to the pipeline:

.. code-block:: python

    from fairseq2.data.parquet.arrow_transform import filter_strings_by_length

    # Create a custom transformation
    def my_transform(table):
        # Apply filtering by text length
        table = filter_strings_by_length(table, "text", min_len=10, max_len=1000)
        return table

    # Apply the transformation
    final_pipeline = loading_pipeline.map(my_transform)

Integration with Hugging Face Datasets
--------------------------------------

fairseq2's parquet dataloader can easily integrate with datasets from the `Hugging Face Hub <https://huggingface.co/datasets>`_ that are available in parquet format. This integration leverages the ``huggingface_hub`` package's ``HfFileSystem`` to seamlessly access parquet files stored on the Hub.

Basic Integration Example
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from fairseq2.data.parquet.fragment_streaming import (
        FragmentStreamingConfig, ParquetFragmentStreamer
    )
    from fairseq2.data.parquet.fragment_loading import (
        FragmentLoadingConfig, ParquetFragmentLoader
    )

    # Initialize the Hugging Face FileSystem
    from huggingface_hub import HfFileSystem
    hf_fs = HfFileSystem()  # FileSystem interface for HF

    # Get dataset files from Hugging Face Hub
    source_dataset_glob_path = "datasets/cais/mmlu/*/*.parquet"
    all_paths = hf_fs.glob(source_dataset_glob_path)  # Find all parquet files
    test_paths = [path for path in all_paths if "test-" in path]  # Optional filtering

    # Configure the fragment streaming
    fragment_config = FragmentStreamingConfig(
        parquet_path=test_paths,
        nb_epochs=1,
        filesystem=hf_fs,  # Provide the Hugging Face filesystem
        split_to_row_groups=True,
        fragment_shuffle_window=0,  # No shuffling in this example
    )

    streamer = ParquetFragmentStreamer(config=fragment_config)

    # Configure the fragment loading
    loading_config = FragmentLoadingConfig(
        columns=None,  # Use all columns
        add_fragment_traces=False,
        drop_null=False,
        nb_prefetch=1,
        num_parallel_fragments=4,
        filters='pc.field("answer") == 0',  # Optional filtering
    )

    # Build the pipeline
    loader = ParquetFragmentLoader(config=loading_config)
    fragment_pipeline = streamer.build_pipeline(0, 1)
    loading_pipeline = loader.apply(fragment_pipeline)

    # Process the results
    tables = list(iter(loading_pipeline.and_return()))

    # Process tables as needed
    # Examples: 
    # - Convert to pandas: df = table.to_pandas()
    # - Convert to polars (efficient): pl.from_arrow(table)

Benefits of Using Hugging Face Datasets with fairseq2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **No Download Required**: Access datasets directly from Hugging Face Hub without manually downloading them first
- **Efficient and Resilient Loading**: Only load the requested dataset, and auto-retry (using ``SafeFragment``) when network issues or expired authentication tokens interrupt the data loading
- **Advanced Processing**: Apply all fairseq2's parquet capabilities (filtering, batching, sharding, etc.)
- **Memory Efficiency**: Stream data without loading entire datasets into memory
- **High Performance**: Leverage the optimized data loading pipeline of fairseq2

This integration is particularly useful for large-scale datasets like multilingual text corpora, embedding collections, or multimodal datasets where efficient data handling is crucial.
