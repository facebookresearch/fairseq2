# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Optional

import pyarrow as pa


@dataclass
class ParquetDatasetLimitOptions:
    """
    Contains different options that allows to load only a part of the provided dataset.
    """

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


@dataclass
class FragmentStreamingConfig:
    """
    This config describes the streaming of fragments from a parquet dataset.
    """

    parquet_path: str | List[str] = str()
    """The path to parquet dataset file.
        if `parquet_path` is remote (like stats with "s3://..."),
        the filesystem will be automatically detected and `filesystem_expr` should remain None
    """

    filesystem: Optional[Any] = None
    """
    A filesystem object or str filesystem expression (`filesystem = eval(filesystem)`)
    that can be used to read the parquet dataset.
    """

    name: Optional[str] = None
    "optional name of the dataset, can be use as a reference for datacard"

    weight: float = 1.0
    """
    Indicates relative weight of dataset that can be used for sampling from different datasets.
    """

    partition_filters: Optional[str | List[str]] = None
    """
    Filters that should be applied only on partition columns for fast partition pruning.
    Filters are passed as a str expression that will be transformed through `eval(...)`.

    If several filters are provided, they will be combined with `AND` operator.

    For concrete syntax of filtering expression, see
    https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Expression.html#pyarrow.dataset.Expression

    Some examples :

    >>> import pyarrow.compute as pc
    >>> import pyarrow as pa

    >>> filters = (pc.field("data_split") == pc.scalar("train")) & (pc.field("duration") > 7)
    >>> filters = pa.compute.greater(pa.compute.utf8_length(ds.field("lang1_text")), 4)
    >>> filters = pa.compute.less_equal(pa.compute.list_value_length(pa.dataset.field("audio_wav")), 16_000 * 30)

    Note that all fields used here should be among existing columns in the dataset schema.


    To know the partition columns on dataset :
    ```python
    >>> pq.ParquetDataset(parquet_path).partitioning.schema.names
    ```
    Note that for if `parquet_path` references a single file -> the result above will NOT be correct (returns all columns).
    Note that for a single file case, there should no partition_filters since there're no partitions !!
    """

    limit: Optional[ParquetDatasetLimitOptions] = None
    """
    Contains different options that allows to load only a part of the provided dataset.
    Good for debugging or testing the data scaling laws.

    It will **always** take some number of **first** fragments according to the order in which
    they appear in the dataset.
    This limit will be applied after `partition_filters` and is not affected by files suffling.

    When several limits are provided, each of them will be applied.
    """

    split_to_row_groups: bool = True
    """If ``True``, uses Parquet row groups instead of simple partitions which
    are generally smaller. Highly recommended for non-partitioned parquet files."""

    seed: int = 123

    fragment_shuffle_window: int = 40
    """
    The number of fragments to shuffle together in streaming fragment reading.
    - If fragment_shuffle_window=-1, the dataset input fragments (row groups and files) will be shuffled globally.
    - If fragment_shuffle_window=0, no shuffling will be applied.

    Larger shuffling window provides a better randomization,
    so, row groups typically will come more uniformly from different files.
    Yet, it requires more time to first batches since splitting to row group requires
    to fetch the meta data from each files.
    The pipeline state is dump and reload will be slower.

    Global shuffle with `fragment_shuffle_window=-1` can be used for relatively small datasets
    (with small number of files) or datasets on NFS where metadata fetching is fast.

    Note `files_circular_shift` is ignored when `fragment_shuffle_window=-1`.
    """

    files_circular_shift: bool = False
    """
    If ``True``, the dataset input files will be shifted in a circular fashion after shuffling
    so that different rank will start to read files stream from different (equality separated) positions.
    Still all files will be read by any of ranks.

    Note also:
    - The files shuffling will be different for each epoch, but the same for all ranks.
    - We use stable fragment hash for sharding so that different ranks will read different fragments.

    It should be for large datasets for pretrained models typically when a single file contains
    a large number of row groups to get an extra randomization.
    """

    nb_epochs: Optional[int] = None
    """
    Number of passes over the data before iterations stop.
    None means infinite number of epochs.
    """

    def __post_init__(self) -> None:
        # TODO: check parameters validity
        pass

    def add_partition_filter(
        self, filters: List[str] | List[pa.dataset.Expression]
    ) -> "FragmentStreamingConfig":
        """
        Add an extra partition filter to the dataset.
        """
        from fairseq2.data.parquet.fragment_streaming.primitives import process_filter

        config = deepcopy(self)

        if config.partition_filters is None:
            config.partition_filters = filters
        elif isinstance(config.partition_filters, str):
            config.partition_filters = process_filter(
                [config.partition_filters] + filters
            )
        elif isinstance(config.partition_filters, list):
            config.partition_filters = process_filter(
                config.partition_filters + filters
            )
        else:
            raise ValueError(
                f"Invalid partition_filters type {type(config.partition_filters)}"
            )
        return config
