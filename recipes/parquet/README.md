## Parquet Data Loading with fairseq2

The recipe module [parquet_dataloader](./parquet_dataloader.py) shows one way to
build an efficient dataloader over a Apache Parquet dataset (partitioned or not)
using `fairseq2.data` primitives. It uses the [pyarrow.parquet](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html)
API to interface with Parquet files, so it requires an extra package
installation with `pip install fairseq2[arrow]`.

The present dataloader is of general purpose and can be combined with various
downstream workflows. Some important technical notes to keep in mind:

* Dataloader will simultaneously load several Parquet dataset fragments
  (`nb_parallel_fragments`) and shuffle their elements together before returning
* Thus, increasing `nb_parallel_fragments` will result in better randomization
  but also increase the memory footprint.
* For heavy rows datasets, prefer save the Parquet files with relatively small
  `row_groups` to improve streaming regularity.
* For reading from S3 storage, `fairseq2.data` being multithreaded,
  `from pyarrow.fs import S3FileSystem` (releasing GIL) works best.
* Currently, only some of pyarrow dtypes are mapped to their torch equivalent,
  this support will improve in the future.

Please refer to the `ParquetBasicDataloaderConfig` for more details about the
existing configuration parameters.

Example of simple usage:

```python
import pyarrow.compute as pc

from recipes.parquet.parquet_dataloader import (
    ParquetBasicDataloaderConfig,
    ParquetBatchFormat,
    build_parquet_iterator_pipeline
)

config = ParquetBasicDataloaderConfig(
    parquet_path="path/to/parquet/dataset", 
    filters=pc.greater(pc.utf8_length(pc.field("src_text")), 5)
    columns=["src_text", "src_lang", "audio_wav"],
    batch_size=20,
    output_format=ParquetBatchFormat.torch,
    world_size=1,
    rank=0,
    seed=123,
)

for batch in parquet_iterator(config):
     pass
```
