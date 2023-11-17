# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import partial

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pyarrow.dataset import get_partition_keys  # requires pyarrow >= 13

from fairseq2.data import CString
from fairseq2.data.data_pipeline import DataPipeline, DataPipelineBuilder, read_sequence


class ParquetBatchFormat(Enum):
    pyarrow = 0
    pandas = 1
    torch = 2


@dataclass  # TODO: (kw_only=True) with python3.10
class ParquetBasicDataloaderConfig:
    parquet_path: str
    """
    Path to parquet dataset file
    """

    batch_size: tp.Optional[int] = None
    """
    Fixed output batch size
    """

    order_by: tp.Optional[str] = None
    """Column in dataset whose value length `L` will be used for batches ordering.
       This results in batches with relatively homogeneous values of `L`,
       typically to support optimal padding.
    """

    max_tokens: tp.Optional[int] = None
    """
    Used with `order_by` option to control the total number of padded tokens in a each batch.
    Typically, this option is preferred to `batch_size` for reducing the memory footprint. 
    """

    columns: tp.Optional[tp.List[str]] = None
    """List of columns to load"""

    filters: tp.Optional[tp.Union[tp.List[tp.Any], pa.dataset.Expression]] = None
    """
    See https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Expression.html#pyarrow.dataset.Expression
    Some examples :

    >>> import pyarrow.compute as pc
    >>> import pyarrow as pa

    >>> filters = [("data_split", "=", "train"), ("lang1", "in", ["eng","spa"]), ("lang2", "=", "eng")])
    >>> filters = (pc.field("data_split") == pc.scalar("train")) & (pc.field("duration") > 7)
    >>> filters = pa.compute.greater(pa.compute.utf8_length(ds.field("lang1_text")), 4)
    >>> filters = pa.compute.less_equal(pa.compute.list_value_length(pa.dataset.field("audio_wav")), 16_000 * 30)

    Note that all fields used here should be among existing columns in the dataset schema
    """

    output_format: ParquetBatchFormat = ParquetBatchFormat.pyarrow
    """
    Format to use for output batches
    """

    split_to_row_groups: bool = True
    """
    Use parquet row groups instead of simple partitions which are generally smaller.
    Highly recommended for non-partitioned parquet file.
    """

    shuffle: bool = True
    """
    Whether to shuffle dataset samples during the iteration.
    If False and `order_by` is None, the batch samples will be produced in natural parquet dataset reading order.
    """

    drop_null: bool = True
    """Dropping rows containing any null value"""

    seed: tp.Optional[int] = None
    """
    seed making iteration deterministic
    """

    min_batch_size: int = 1
    """Drops batches whose length < `min_batch_size`"""

    nb_producers: int = 5
    """Number of parquet partitions read allowed to be read symonymously.
       Higher values will result in higher speed, better randomization and higher memory footprint.
       If partitions size is rather small compared to batch size, we recommend to increase nb_producers.
    """

    nb_prefetch: int = 2
    """
    Nb of producers groups (of size `nb_producers`) to prefetch
    """

    rank: int = 0
    """The rank of this worker in the process group."""

    world_size: int = 1
    """The world size of the process group."""

    num_parallel_calls: int = 8
    """The number of parallel calls in map operations."""

    use_threads: bool = False
    """
    Whether pyarrow should use its internal parallelism threads to read the parquet part.
    Since we rely on the external parallelism, this param is tuned off.
    """

    def __post_init__(self) -> None:
        assert self.parquet_path, "requires path"
        assert self.num_parallel_calls >= 1
        assert self.world_size >= 1
        assert self.world_size - 1 >= self.rank >= 0
        assert self.nb_prefetch >= 1
        assert self.nb_producers >= 1

        assert self.min_batch_size >= 0
        assert self.batch_size is None or self.batch_size > 0
        assert self.max_tokens is None or self.max_tokens > 0

        if not ((self.batch_size is None) ^ (self.max_tokens is None)):
            raise ValueError("need to provide either `batch_size` either `max_tokens`")
        if self.max_tokens is not None and self.order_by is None:
            raise ValueError("`order_by` should be given to deal with `max_tokens`")

        if self.filters is not None and not isinstance(
            self.filters, pa.dataset.Expression
        ):
            self.filters = pq.filters_to_expression(self.filters)


@contextmanager
def pyarrow_cpu(nb_cpu: int) -> tp.Generator[None, None, None]:
    nb_cpu_old = pa.cpu_count()
    nb_io_cpu_old = pa.io_thread_count()
    pa.set_cpu_count(nb_cpu)
    pa.set_io_thread_count(nb_cpu)
    try:
        yield
    finally:
        pa.set_cpu_count(nb_cpu_old)
        pa.set_io_thread_count(nb_io_cpu_old)


NestedDict = tp.Dict[str, "NestedDictValue"]
NestedDictValue = tp.Union[torch.Tensor, tp.List[CString], pd.Series, NestedDict]

BatchOutputType = tp.Union[pa.Table, pd.DataFrame, NestedDict]


def from_pyarrow_to_torch_tensor(
    arr: tp.Union[pa.Array, pa.ChunkedArray], strict: bool = True
) -> NestedDictValue:
    """
    struct_array = pa.Array.from_pandas([{"x": 4, "y": "RR"}] * 10)
    nest_array = pa.Array.from_pandas([[{'a': 1}, {'a': 2}]])
    """
    # for future ideas https://arrow.apache.org/docs/python/generated/pyarrow.Tensor.html
    # for sparse matrix support https://github.com/apache/arrow/blob/main/python/pyarrow/tests/test_sparse_tensor.py

    assert arr.null_count == 0, "does not support null values yet"

    if isinstance(arr, pa.ChunkedArray):
        arr = arr.chunks[0] if arr.num_chunks == 1 else arr.combine_chunks()

    arr_type = arr.type
    if pa.types.is_primitive(arr_type):
        return torch.from_numpy(arr.to_numpy(zero_copy_only=True))

    try:
        return torch.from_numpy(arr.to_numpy(zero_copy_only=True))
    except pa.ArrowInvalid:
        pass

    if pa.types.is_dictionary(arr_type):
        return from_pyarrow_to_torch_tensor(arr.dictionary_decode())

    if pa.types.is_string(arr_type):
        return list(map(CString, arr.to_pandas()))

    if (
        pa.types.is_list(arr_type) or pa.types.is_large_list(arr_type)
    ) and pa.types.is_primitive(arr_type.value_type):
        return torch.nested.as_nested_tensor(
            list(map(torch.from_numpy, arr.to_pandas()))
        )

    if pa.types.is_fixed_size_list(arr_type) and pa.types.is_primitive(
        arr_type.value_type
    ):
        return torch.from_numpy(np.reshape(arr.values, (-1, arr_type.list_size)))

    if pa.types.is_struct(arr_type):
        return {
            arr_type.field(i).name: from_pyarrow_to_torch_tensor(arr.field(i))
            for i in range(arr_type.num_fields)
        }

    if pa.types.is_nested(arr_type):
        # TODO: deal with arr = [[{'a': 1}, {'a': 2}]]
        pass

    if strict:
        raise NotImplementedError(f"{arr_type} cannot be converted to torch.Tensor")
    else:
        return arr.to_pandas()


def pyarrow_table_to_torch_dict(tt: pa.Table, strict: bool = True) -> NestedDict:
    return {
        col: from_pyarrow_to_torch_tensor(tt[col], strict) for col in tt.column_names
    }


class _TableWrapper:
    """
    class to avoid fairseq2 casting pa.Table to iterable objects
    which currently fails
    """

    def __init__(self, table: pa.Table) -> None:
        self.table: pa.Table = table


class ParquetBasicDataLoader:
    """
    Example of usage :

       >>> from fairseq2.utils.parquet_dataloader import ParquetBasicDataLoader
       >>> from tqdm.auto import tqdm
       >>> bpd_config = BasicParquetDataloaderConfig(parquet_path="...", batch_size=20,
       ...                                           columns=["src_text", "src_lang", "audio_wav"],
       ...                                           output_format=ParquetBatchFormat.torch)
       >>> pq_dl = ParquetBasicDataLoader(bpd_config)
       >>> ei_batch = iter(pq_dl)
       >>> res = []
       >>> for i, batch in tqdm(enumerate(ei_batch)): res.append(len(batch))

    """

    config: ParquetBasicDataloaderConfig
    source_ds: pq.ParquetDataset
    _epoch: int = 0

    def __init__(
        self,
        config: ParquetBasicDataloaderConfig,
    ) -> None:
        self.config = config

        # split_row_groups=True is not supported yet
        self.source_ds = pq.ParquetDataset(
            self.config.parquet_path, validate_schema=True, filters=self.config.filters
        )

        self.columns = self.config.columns or self.source_ds.schema.names
        assert set(self.columns).issubset(set(self.source_ds.schema.names))

        if self.config.order_by is not None:
            assert self.config.order_by in self.source_ds.schema.names

        partitioning_keys = (
            [
                name
                for (name, dd) in zip(
                    self.source_ds.partitioning.schema.names,
                    self.source_ds.partitioning.dictionaries,
                )
                if dd is not None
            ]
            if self.source_ds.partitioning
            else []
        )
        columns_wo_partition_keys = [
            col for col in self.columns if col not in partitioning_keys
        ]

        if self.config.order_by is not None:
            self._columns_to_read = sorted(
                set(columns_wo_partition_keys) | set([self.config.order_by])
            )
        else:
            self._columns_to_read = columns_wo_partition_keys

        if self.config.split_to_row_groups:
            self._all_fragments = [
                piece
                for fragment in self.source_ds._dataset.get_fragments(
                    self.config.filters
                )
                for piece in fragment.split_by_row_group()
            ]
        else:
            self._all_fragments = list(
                self.source_ds._dataset.get_fragments(self.config.filters)
            )

    # def __repr__(self) -> str:
    #     """ count rows and other simple stats """"

    # def head(self, nb:int=5) -> pa.Table:
    #     return self.source_ds.head(nb=5)

    def schema(self) -> pa.Schema:
        """
        Returns the full schema of the parquet datasource
        """
        return self.source_ds.schema

    @staticmethod
    def _add_partitioning_values(
        table: pa.Table, fragment: pa.dataset.Fragment
    ) -> pa.Table:
        for key, val in get_partition_keys(fragment.partition_expression).items():
            values = pd.Series([val] * len(table), dtype="category")
            table = table.append_column(key, pa.Array.from_pandas(values))
        return table

    @staticmethod
    def _sanitize_dict(table: pa.Table) -> pa.Table:
        for i, (name, type) in enumerate(zip(table.schema.names, table.schema.types)):
            if pa.types.is_dictionary(type):
                ca = table[name].cast(type.value_type)
                table = table.remove_column(i)
                table = table.append_column(name, ca)
        return table

    @staticmethod
    def compute_length_splits(
        length_col: npt.NDArray[np.int32], max_tokens: int
    ) -> tp.List[npt.NDArray[np.int32]]:
        """split sequence of length_col in the chunks such that total length is ~ max_tokens
           countint the padding to max length of elements in a chunk

        Args:
            length_col (np.ndarray):
            max_tokens (int):

        Returns:
            tp.List[np.ndarray]: splits that contain indices over the original length_col
        """
        argsort_ind = np.argsort(length_col)
        # TODO: remove 0 lengths
        sorted_length_col = length_col[argsort_ind]

        splits = []
        ptr = 0
        for i, length in enumerate(sorted_length_col):
            if length * (i - ptr) > max_tokens:
                splits.append(argsort_ind[ptr : (i - 1)])
                ptr = i - 1
        if (
            length <= max_tokens
        ):  # we drop the last iteration if it results in a batch greater than max_tokens
            splits.append(argsort_ind[ptr:])
        return splits

    @staticmethod
    def compute_length(pa_array: pa.Array) -> npt.NDArray[np.int32]:
        type_ = pa_array.type
        if pa.types.is_list(type_) or pa.types.is_large_list(type_):
            length_col = pa.compute.list_value_length(pa_array).to_numpy()
        elif pa.types.is_string(type_):
            length_col = pa.compute.utf8_length(pa_array).to_numpy()
        else:
            length_col = np.asarray(pa_array.to_pandas().apply(len))

        length_col = length_col.copy()
        length_col[np.isnan(length_col)] = 0
        return np.asarray(length_col, dtype=np.int32)

    @staticmethod
    def concat_table(
        list_table: tp.List[_TableWrapper], drop_null: bool
    ) -> _TableWrapper:
        return _TableWrapper(
            pa.concat_tables(
                [tt.table.drop_null() if drop_null else tt.table for tt in list_table]
            ).combine_chunks()
        )

    def _load_one_fragement(self, fragment: pa.dataset.Fragment) -> _TableWrapper:
        fragment_table = fragment.to_table(
            columns=self._columns_to_read, use_threads=self.config.use_threads
        )
        fragment_table = self._add_partitioning_values(fragment_table, fragment)
        if self.config.filters is not None:
            fragment_table = fragment_table.filter(self.config.filters)
        return _TableWrapper(fragment_table)

    def build_iterator_over_concat_table(
        self, wrap_table: _TableWrapper, random_state: np.random.RandomState
    ) -> DataPipeline:
        order_by = self.config.order_by
        batch_size = self.config.batch_size
        max_tokens = self.config.max_tokens

        table: pa.Table = wrap_table.table
        if order_by is not None:
            length_col = self.compute_length(table[order_by])
            # add small perturbation to avoid same sample appear together during different epochs
            if self.config.shuffle:
                perturbation = random_state.randint(
                    0,
                    np.quantile(length_col, 0.001).astype(np.int32) + 2,
                    len(length_col),
                )
                length_col += np.asarray(perturbation, dtype=np.int32)
        else:
            if self.config.shuffle:
                length_col = random_state.randint(0, 2**23, len(table))
            else:
                length_col = np.zeros(len(table), dtype=np.int32)

        table = table.select(self.columns)

        if batch_size is not None:
            order_tt = pa.Table.from_arrays(
                [pa.array(np.argsort(length_col, kind="stable"))], ["order"]
            )
            batches = [ind["order"] for ind in order_tt.to_batches(batch_size)]
        elif max_tokens is not None:
            batches = self.compute_length_splits(length_col, max_tokens)
        else:
            raise ValueError("unknown batching method")

        if self.config.shuffle:
            batches = [batches[i] for i in random_state.permutation(len(batches))]

        return (
            read_sequence(batches)
            .map(
                lambda ind: _TableWrapper(table.take(ind).combine_chunks()),
                num_parallel_calls=max(self.config.num_parallel_calls // 2, 1),
            )
            .and_return(max_num_warnings=4)
        )

    def build_epoch_iterator_pipeline(
        self, seed: tp.Optional[int] = None, epoch: int = 0
    ) -> DataPipelineBuilder:
        seed = seed if seed is not None else self.config.seed
        np_rs = np.random.RandomState(
            hash((seed, epoch)) % 2**32
        )  # TODO: use stable hashing instead python
        if self.config.shuffle:
            all_shuffled_fragments = list(np_rs.permutation(self._all_fragments))
        else:
            all_shuffled_fragments = self._all_fragments

        pipeline_builder = (
            read_sequence(all_shuffled_fragments)
            .shard(shard_idx=self.config.rank, num_shards=self.config.world_size)
            .map(
                self._load_one_fragement,
                num_parallel_calls=self.config.num_parallel_calls,
            )
            .bucket(self.config.nb_producers)
            .prefetch(self.config.nb_prefetch)
            .map(
                partial(self.concat_table, drop_null=self.config.drop_null),
                num_parallel_calls=self.config.nb_prefetch,
            )
            .yield_from(
                partial(self.build_iterator_over_concat_table, random_state=np_rs)
            )
            .filter(lambda wt: bool(len(wt.table) >= self.config.min_batch_size))
        )

        if self.config.output_format == ParquetBatchFormat.pandas:
            pipeline_builder = pipeline_builder.map(
                lambda wt: _TableWrapper(wt.table.to_pandas())
            )
        elif self.config.output_format == ParquetBatchFormat.torch:
            pipeline_builder = pipeline_builder.map(
                lambda wt: pyarrow_table_to_torch_dict(wt.table)
            )
        return pipeline_builder

    def __iter__(self) -> tp.Generator[BatchOutputType, None, None]:
        def _to_real_object(x: tp.Union[_TableWrapper, NestedDict]) -> BatchOutputType:
            if isinstance(x, _TableWrapper):
                return x.table
            else:
                return x

        with pyarrow_cpu(self.config.num_parallel_calls):
            yield from map(
                _to_real_object,
                iter(
                    self.build_epoch_iterator_pipeline(self.config.seed, self._epoch)
                    .prefetch(self.config.num_parallel_calls)
                    .and_return(max_num_warnings=4)
                ),
            )
        self._epoch += 1
