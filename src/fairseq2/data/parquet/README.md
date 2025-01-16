## Parquet Data Loading with fairseq2

The recipe module [pipeline](./pipeline.py) shows how to build an efficient dataloader over an Apache Parquet dataset (partitioned or not) using `fairseq2.data` primitives. It uses the [pyarrow.parquet](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html) API to interface with Parquet files, so it requires an extra package installation with `pip install fairseq2[arrow]`.

The present dataloader is of general purpose and can be combined with various downstream workflows. Some important technical notes to keep in mind:

* Dataloader will simultaneously load several Parquet dataset fragments (`nb_parallel_fragments`) and shuffle their elements together before returning
* Thus, increasing `nb_parallel_fragments` will result in better randomization but also increase the memory footprint
* For heavy rows datasets, prefer saving the Parquet files with relatively small `row_groups` to improve streaming regularity
* For reading from S3 storage, `fairseq2.data` being multithreaded, `from pyarrow.fs import S3FileSystem` (releasing GIL) works best
* Currently, only some of pyarrow dtypes are mapped to their torch equivalent, this support will improve in the future

The dataloader configuration is split into two parts:
- `ParquetDatasetConfig`: Configures the dataset source and filtering
- `ParquetBasicDataloaderConfig`: Controls the data loading behavior like batching and shuffling

Example of simple usage:

```python
import pyarrow.compute as pc

from fairseq2.data.parquet import (
    ParquetBasicDataloaderConfig,
    ParquetDatasetConfig,
    ParquetBatchFormat,
    parquet_iterator,
)

# Configure the dataset source
dataset_config = ParquetDatasetConfig(
    parquet_path="path/to/parquet/dataset",
    filters=pc.greater(pc.utf8_length(pc.field("src_text")), 5),
    split_to_row_groups=True,
    nb_parallel_fragments=5
)

# Configure the data loading behavior
dataloader_config = ParquetBasicDataloaderConfig(
    batch_size=20,
    output_format=ParquetBatchFormat.torch,
    shuffle=True,
    world_size=1,
    rank=0,
    seed=123,
)

# Iterate over batches
for batch in parquet_iterator(dataset_config, dataloader_config):
    pass
```

Please refer to the `ParquetDatasetConfig` and `ParquetBasicDataloaderConfig` classes for more details about the available configuration parameters.
