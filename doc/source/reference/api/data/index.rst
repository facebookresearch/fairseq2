.. _data:

fairseq2.data
=============

.. currentmodule:: fairseq2.data

The data module provides flexible data processing pipelines and utilities for working
with various data formats including text, audio, and structured data. It includes
high-performance data loaders, text tokenization systems, and specialized processing
utilities for machine learning workflows.

**Key Features:**

- **High-Performance Data Pipelines**: Optimized C++-based data loading and processing
- **Text Processing**: Comprehensive tokenization and text preprocessing utilities
- **Audio Processing**: Tools for audio data loading and feature extraction
- **Structured Data**: Support for Parquet, JSON, and other structured formats
- **Memory Efficient**: Streaming and batched processing for large datasets
- **Integration**: Seamless integration with fairseq2's training and evaluation systems

.. note::
   The fairseq2 data module is designed for high-throughput machine learning workloads
   and provides both Python and C++ implementations for performance-critical operations.

Text Processing
---------------

.. toctree::
   :maxdepth: 1

   tokenizer

Data Pipeline Components
------------------------

**Coming Soon**: Documentation for data pipeline components including:

- **DataPipeline**: Core pipeline abstraction for chaining data transformations
- **DataLoader**: High-performance data loading with batching and shuffling
- **Collators**: Utilities for batching variable-length sequences
- **Samplers**: Various sampling strategies for training and evaluation

Audio Processing
----------------

**Coming Soon**: Documentation for audio processing utilities including:

- **AudioDataset**: Dataset classes for audio files and features
- **AudioCollator**: Batching utilities for variable-length audio sequences
- **Feature Extraction**: Mel-spectrogram, MFCC, and other audio features

Structured Data
---------------

**Coming Soon**: Documentation for structured data processing including:

- **ParquetDataset**: Efficient loading of Parquet files
- **JsonDataset**: JSON and JSONL file processing
- **CSVDataset**: CSV file loading with type inference

See Also
--------

* :doc:`../nn/index` - Neural network components and BatchLayout
* :doc:`../datasets` - High-level dataset abstractions
* :doc:`../models` - Model implementations that consume data
