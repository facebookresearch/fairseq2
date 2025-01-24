# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq


class ParquetBatchFormat(Enum):
    pyarrow = 0
    pandas = 1
    torch = 2


@dataclass
class ParquetDatasetLimitOptions:
    """
    Contains different options that allows to load only a part of the provided dataset.
    """

    columns: Optional[List[str]] = None
    """The list of columns to load."""

    fraction_of_files: Optional[float] = None
    """
    Load only a fraction of the provided dataset.
    """

    nb_files: Optional[int] = None
    """
    Number of files to load.
    """

    nb_fragments: Optional[int] = None
    """
    Number of fragments to load.
    """

    nb_rows: Optional[int] = None
    """
    Number of rows to load.
    """

    limit_nb_tokens: Optional[int] = None
    """
    Number of tokens to load.
    """

    token_columns: Optional[List[str]] = None
    """
    Columns to use for token counting.
    """


@dataclass
class ParquetDatasetConfig:
    """
    Config for datasets stored in Parquet format.

    All None value should be filled up in downstream `build_parquet_iterator_pipeline`.
    """

    name: Optional[str] = None
    """When name is provided, it will use preregistered cards to populate all attributes.
        name convention is the following
        -  {card_name}={split}:{weight}

        Example:
        - wiki
        - wiki:0.2 # no split
        - wiki=dev  # default weight=1
        - wiki=dev:0.2

        Cards attributes will be overwritten by user defined ParquetDatasetConfig in
        `create_dataset_config_from_cards`.
    """

    parquet_path: str = ""
    """The path to parquet dataset file.
    """

    weight: float = 1.0
    """
    Indicates relative weight of dataset that can be used for sampling from different datasets.
    """

    limit: Optional[ParquetDatasetLimitOptions] = None
    """
    Contains different options that allows to load only a part of the provided dataset.
    It will **always** take some number of **first** fragments according to the order in which
    they appear in the dataset and this logic will not be depedent on suffling/seed.
    When several limits are provided, each of them will be applied (resulting in the strongest limit).
    """

    partition_filters: Optional[str] = None
    """
    Filters that should be applied only on partition columns for fast partition prunning.
    This filters should not be duplicated in `filters` (below) which are used on materialized data.
    To know the partition columns on dataset :
    ```python
    >>> pq.ParquetDataset(parquet_path).partitioning.schema.names
    ```
    Note that for if `parquet_path` references a single file -> the result above will NOT be correct (returns all columns).
    Note that for a single file case, there should no partition_filters since there're no partitions !!
    """

    filters: Optional[Union[List[Any], pa.dataset.Expression]] = None
    """
    This can be any valid pyarrow dataset filter expression, or a list of old-style tuples
    that we convert to an expression in __post_init__.

    See https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Expression.html#pyarrow.dataset.Expression

    Some examples :

    >>> import pyarrow.compute as pc
    >>> import pyarrow as pa

    >>> filters = [("data_split", "=", "train"), ("lang1", "in", ["eng","spa"]), ("lang2", "=", "eng")])
    >>> filters = (pc.field("data_split") == pc.scalar("train")) & (pc.field("duration") > 7)
    >>> filters = pa.compute.greater(pa.compute.utf8_length(ds.field("lang1_text")), 4)
    >>> filters = pa.compute.less_equal(pa.compute.list_value_length(pa.dataset.field("audio_wav")), 16_000 * 30)

    Note that all fields used here should be among existing columns in the dataset schema.
    """

    split_to_row_groups: bool = True
    """If ``True``, uses Parquet row groups instead of simple partitions which
    are generally smaller. Highly recommended for non-partitioned parquet files."""

    nb_parallel_fragments: Optional[int] = None
    """
    This parameter can be dataset specific:
    For dataset with large number of sentences per document (sample),
    it's enough to set `nb_parallel_fragments=2 or 3`.
    For datasets, with smaller number of sentences (~10) and small row_group_size (~200-600),
     `nb_parallel_fragments` could be increase to 10 - 20.

    The number of Parquet fragments allowed to be read in parallel. Higher
    values will result in higher speeds, better randomization, and higher memory
    footprint. If partition size is rather small compared to the batch size, we
    recommend to increase ``nb_parallel_fragments``.

    Leaving ``nb_parallel_fragments`` to None will trigger auto-detection based on dataset metadata.
    """

    def __post_init__(self) -> None:
        if not self.parquet_path:
            raise ValueError(f"requires non-empty path got {self.parquet_path}")

        if self.filters is not None and not isinstance(
            self.filters, pa.dataset.Expression
        ):
            self.filters = pq.filters_to_expression(self.filters)


@dataclass
class DataLoadingConfig:
    multiple_dataset_chaining: str = "sample"
    """
    This option allows to chain several datasets together.
    The chaining can be done in two ways:
    - `sample` : each dataset will be sampled with the provided weight
    - `concat` : datasets will be concatenated together (no weights taken into account)
    - `round_robin`: datasets will be sampled in a round robin fashion (no weights taken into account)
    """
    batch_size: Optional[int] = None
    """The output batch size."""

    order_by_length: Optional[bool] = True
    """
    Whether to create the batches with homogeneous tokens length
    for more efficient padding.
    """

    max_tokens: Optional[int] = None
    """Used with the ``order_by_length`` option to control the total number of
    padded tokens in each batch. Typically, this option is preferred over
    ``batch_size`` to reduce the memory footprint.
    """

    len_to_wrap_long_seq: Optional[int] = None
    """
    Wrapping a source sequences to the length of `len_to_wrap_long_seq`.
    For instance, for a `len_to_wrap_long_seq=2`
    batch = {
        "source": [["v1", "v2", "v3", "v4", "v5"], ["u1", "u2", "u3"], ["w1"]],
    }
    will be transormed to
    1. if packing is False :
    batch = {
        "source": [['v1', 'v2'], ['v3', 'v4'], ['v5'], ["u1", "u2"], ["u3"], ["w1"]]
    }
    1. if packing is True :
    batch = {
        "source": [['v1', 'v2'], ['v3', 'v4'], ['v5', 'u1'], ["u2", "u3"], ["w1"]]
    }

    Note: currently only allowed to be used with no "target" provided (unsupervised style) !
    """

    packing: bool = False
    """
    If True, all sequential documents (seqs of sentences) will be concated into one big document
    before applying wrapping.
    This will result in all samples (except maybe one) having exactly `len_to_wrap_long_seq` length !
    """

    wrap_before_affixing: bool = False
    """
    If True, we will wrap the sequences before adding the source prefix/suffix.
    Recommended when pre-training with packed data i.e len_to_wrap_long_seq not None and packing=True
    """

    output_format: ParquetBatchFormat = ParquetBatchFormat.torch
    """The format to use for output batches."""

    shuffle: bool = True
    """If ``True``, shuffles the dataset samples during the iteration. If ``False``
    and ``order_by_length`` is ``None``, the batch samples will be produced in
    natural Parquet dataset reading order."""

    drop_null: bool = True
    """If ``True``, drops rows containing any null value."""

    seed: int = 123
    """The RNG seed value for deterministic behavior."""

    nb_epochs: int = 100
    """
    Number of passes over the data before iterations stop
    """

    min_batch_size: int = 1
    """Drops batches whose length is less than ``min_batch_size``"""

    nb_prefetch: int = 3
    """The number of producer groups (of size `nb_parallel_fragments`) to
    prefetch."""

    world_size: int = 1
    """The world size of the process group."""

    rank: int = 0
    """The rank of this worker in the process group."""

    num_parallel_calls: int = 2
    """The number of parallel calls in map operations."""

    use_threads: bool = False
    """Whether pyarrow should use its internal threads to read the Parquet file.
    Since we rely on the external parallelism, this param is tuned off by
    default."""

    ignore_checkpointed_pipeline: bool = False
    """Whether to ignore the saved datapipeline state or load it when resuming.
    Temporary fix for issues re-loading saved checkpoints"""

    sharding_in_memory: bool = False
    """
    If True, the dataset will be sharded in memory.
    """

    even_sharding: bool = False
    """
    This option should be activated ONLY for validataion on small datasets
    to guarantee the perfect data sharding accross the workers.
    Note that in current impmentation, activating `even_sharding` requires `sharding_in_memory=True`
    which will lead to big overhead for big dataset.
    Note also that some fraction of the data may be dropped due to even sharding.
    For big validation datasets, prefer using large `nb_epoch` + limiting `max_validation_iterations`
    instead of using `even_sharding` !

    For training use case, it should left to False and combined with large number of epochs.
    For evaluation use case, it also should be False since we dont care about the batch syncronization across different workers.
    """
    max_iteration_steps: Optional[int] = None
    """
    If not None, it will be used to limit the number of batches produced per each dataset
    """

    def __post_init__(self) -> None:
        if not ((self.batch_size is None) ^ (self.max_tokens is None)):
            raise ValueError("need to provide either `batch_size` either `max_tokens`")
        if self.max_tokens is not None and self.order_by_length is None:
            raise ValueError(
                "`order_by_length` should be given to deal with `max_tokens`"
            )
        if not self.sharding_in_memory and self.even_sharding:
            raise ValueError("`even_sharding` requires `sharding_in_memory=True`")


@dataclass
class ValidationDataLoadingConfig(DataLoadingConfig):
    """
    This class allows to have some hardcoded parameters for data loading of validation datasets
    """

    multiple_dataset_chaining: str = "concat"
    nb_epochs: int = 1
    min_batch_size: int = 1  # we want to keep all samples
    shuffle: bool = False  # we dont need the randomness here
    batch_size: Optional[int] = 10
    max_tokens: Optional[int] = None
    """
    Leaving both `max_tokens` and `batch_size` to None will trigger
    auto-detection based on dataset metadata and distributed training world size
    to make more or less even distribution of samples across workers.
    Typically, if worker_batch_size = total_batch_size // world_size <= 40,
    we will use batch_size=worker_batch_size, otherwise we will use
    max_tokens=min(total_tokens_number // world_size, 3000).
    See dataloading:SingleParquetDatasetDataloader::set_validation_params for more details.
    """


@dataclass
class EvaluationDataLoadingConfig(DataLoadingConfig):
    """
    This class allows to have some hardcoded parameters for data loading of evaluation datasets.
    In partitcular, even in distributed setup evaluation should not require workers syncronization.
    Therefore, we set `even_sharding` = False to get the all data samples !
    """

    multiple_dataset_chaining: str = "concat"
    nb_epochs: int = 1  # only ONE full pass over the full data !
    min_batch_size: int = 1  # we want to keep all samples
    shuffle: bool = False  # we dont need the randomness here
    batch_size: Optional[int] = 10
    max_tokens: Optional[int] = None  # this should be ok for most of models
    even_sharding: bool = False  # we dont want to lose any sample !
    sharding_in_memory: bool = True  # activate sharding by rank and world size
    max_samples: Optional[int] = None
    """evaluate only the first n samples (for debugging)"""


@dataclass
class ParquetBasicDataloaderConfig(DataLoadingConfig):
    """
    Parquet-specific data loading config that extends the generic DataLoadingConfig.
    """

    output_format: ParquetBatchFormat = ParquetBatchFormat.pyarrow
    """The format to use for output batches."""
