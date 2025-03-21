## Parquet Data Loading with fairseq2

The subfolder contains some tooling necessary to build an efficient dataloader over an Apache Parquet dataset (partitioned or not) in `fairseq2.data`.
The present tooling is of general purpose and can be combined with various downstream workflows.
Note that it requires an extra package installation with `pip install fairseq2[arrow]` since we rely on the [pyarrow](https://arrow.apache.org/docs/python/index.html) library to interface with parquet files.


Folder is organized as follows:
* [fragement_streaming](fragment_streaming/builder.py): is responsible for building various schedulers that produce Parquet dataset fragments;
* [fragement_loading](fragment_loading/builder.py): is responsible for the data reading from the Parquet dataset fragments;
* [table_bucketing](table_bucketing/builder.py): contains the logic of how to bucketize loaded data several loaded tables in memory.


## Fragments streaming

We're using the [pyarrow.parquet](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html) API to interface with Parquet datasets.
A parquet dataset is a collection of parquet files that can be partitioned or not. Each parquet file is a collection of row groups.
And, roughly speaking, a row group is the smallest piece of parquet file that can be read in memory. Yet since parquet is columnar format, we can always read only a subset of columns from a row group.


`fragment_streaming/config.py:FragmentStreamingConfig` defines the configuration of the "fragment" streaming pipeline.
A fragment corrreponds to a file if the `split_to_row_groups` is set to `False` and to a row group otherwise.
For better streaming performance, we recommend to set `split_to_row_groups` to `True` which results in smaller fragments in memory.

### on shuffling
If `fragment_shuffle_window` is set to `0`, no shuffling will be applied.
if `fragment_shuffle_window` is set to `-1`, all fragments will be shuffled globally.

For other values of `fragment_shuffle_window`, shuffle is enabled and it will happen as follows.
All files dataset will be shuffled globally (and this shuffling will be different from one epoch to another).
Next, each file will be split into row groups and shuffled locally within `fragment_shuffle_window`.

Note that the global shuffling requires to get parquet all files metadata upfront which can be expensive for large datasets located remotely, whearas  if `fragment_shuffle_window` is set to a small value (e.g. ~ average nb fragments per file * 5), the  time to the first batch will be faster. The metadata will be fetching will be done on the fly in that case.

Note also that shuffling behavior is seeded to be completely deterministic by the `seed` parameter, thus if one reset a pipeline with the same `seed` value, the exactly same shuffling will be applied.


### Example of simple usage:
```python
import pyarrow as pa
import pyarrow.compute as pc
from fairseq2.data.parquet import *

fragment_config = FragmentStreamingConfig(
    parquet_path="path/to/parquet/dataset_root/",
    nb_epochs=2,
    split_to_row_groups=True,  # working with row groups instead of files
    fragment_shuffle_window=-1, # -1 means global shuffle
)
fragement_pipeline = ParquetFragmentStreamer(config=fragment_config).build_pipeline(rank=3, world_size=8)

```
Here, `fragement_pipeline` will produce `pa.dataset.Fragment` objects which are kind of pointer to the physical data location that we will read from.

### on sharding
Here we can shard a dataset at fragment level using the `rank` and `world_size` parameters in `build_pipeline`.
This sharding will typically be uneven in terms of resulting number of rows, so we recommend to use the `nb_epochs=None` for inifinte loop for large training runs. Alternatively, if parquet dataloading is not bottleneck, one can stream all fragments without sharding, load them in memory and only then shard them at row level to get more uniform sharding.


### on filtering
One can use the `partition_filters` (as `pa.compute.Expression`) parameter to restrict the dataset on a subset of the parquet files.
Example:
* `partition_filters='pc.field("ds") == "2023-01-01"'`
* `partition_filters='pc.is_in(pc.field("split"), pa.array(["dev", "test"]))'`


## Fragments loading

`fragment_loading/config.py:FragmentLoadingConfig` defines the configuration of the "fragment" loading pipeline.
In particular it uses `NamedColumns` dataclass abstraction to define the columns to read and their renaming.
It's useful to unify the column names across the different datasets with different schemas.


```python

@dataclass
class MyColumnsSchema(NamedColumns):
    category: str
    uid: str
    extra_columns: List[str]


loading_config = FragmentLoadingConfig(
    columns=MyColumnsSchema(category="cat", uid="id", extra_columns=["seq"]),
    add_fragment_traces=True, # adding extra columns to trace the data origin (file path, row group id, row index)
    num_parallel_fragments=4,  # reading 4 fragments in parallel
    nb_prefetch=1,
)

loading_pipeline = ParquetFragmentLoader(config=loading_config).build_pipeline(fragement_pipeline)
```

In the above example, `loading_pipeline` will produce `pa.Table` objects.
Note that raw column `cat` will be renamed to `category`, `id` to `uid`, and `seq` will be kept as is!
If we set `columns=None`, all available columns will be read without any renaming.


## Table bucketing
We can bucketize or recombine several consecutive loaded tables into a a different shape using
`table_bucketing/config.py:TableBucketingConfig`.

```python
    bucketing_config = TableBucketingConfig(target_table_size=1000,
                                            min_fragment_number=2,
                                            max_fragment_number=10,
                                            shuffle=True,
                                            batch_size=5)
    bucketing_pipeline = TableBucketer(bucketing_config).build_pipeline(loading_pipeline)
```
The configuration above will
1. take between 2 and 10 consecutive loaded tables (the precise number is determined in such a way that the total size is just above 1000 rows)
2. concatenate them together and shuffle them in memory
3. then split it into batches of size 5 that will be yielded one by one.


## Putting everything together
One can combine the steps above into a basic yet complete pipeline using `build_basic_parquet_data_pipeline`:

```python
    >>> from fairseq2.data.parquet import *
    >>> config = BasicDataLoadingConfig(
    ...     fragment_stream_config=FragmentStreamingConfig(
    ...         parquet_path="path/to/parquet/dataset/",
    ...         partition_filters='pc.field("split") == "train"',
    ...         nb_epochs=None,
    ...         fragment_shuffle_window=100),
    ...     fragment_load_config=FragmentLoadingConfig(columns=None, nb_prefetch=2, num_parallel_fragments=3),
    ...     table_bucketing_config=TableBucketingConfig(target_table_size=1000,
    ...                                                 min_fragment_number=2, max_fragment_number=10,
    ...                                                 shuffle=True, batch_size=5),
    ... )
    >>> pipeline = build_basic_parquet_data_pipeline(config).and_return()
    >>> for batch in pipeline:
    ...     print(batch.to_pandas())
```


## Pyarrow Table converion

Parquet fragments are represented in memory as [`pa.Table` objects](https://arrow.apache.org/docs/python/generated/pyarrow.Table.html),
one can convert them into other format using the following:
* `pa.Table.to_pandas()` to convert into pandas dataframe;
* `pa.Table.to_pydict()` to convert into a dictionary of lists;
* using [polars](https://docs.pola.rs/) one can use `pl.from_arrow(pa_table, rechunk=False)` to convert into a polars dataframe (with almost memory zero copy);
* `pa.Table.to_pylist()` or `pl.from_arrow(...).to_dicts()` (usually much faster) to convert into a list of dictionaries;
* `parquet/utiles.py:pyarrow_table_to_torch_dict` to convert pyarrow table into a dictionary of cpu torch tensors (best effort).


Note that both pyarrow and polars offers powerful APIs to manipulate and transform tables objects that can be used out of the box with the present tooling.
