# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pickle import dumps, loads
from typing import List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from retrying import retry

from fairseq2.data.parquet.arrow_transform.transform import (
    add_fragments_trace,
)
from fairseq2.data.parquet.utils import (
    add_partitioning_values,
    fragment_stable_hash,
)
from fairseq2.logging import log

loading_retry = retry(
    retry_on_exception=lambda exception: isinstance(exception, OSError),
    stop_max_attempt_number=1,
    wait_exponential_multiplier=2,
    wait_exponential_max=20,
)


def init_parquet_dataset(
    parquet_path: str | List[str],
    partition_filters: Optional[pa.dataset.Expression] = None,
    filesystem=None,
) -> pq.ParquetDataset:
    """
    Initialize a Parquet dataset.
    Leaving `filesystem` to None will trigger the detection of the filesystem.

    Args:
        parquet_path (str or List[str]): The path to the Parquet dataset.
        filters (Optional[pa.dataset.Expression]): Partition level filters to apply to the dataset.
        filesystem : The filesystem to use. If None, the filesystem will be detected.

    Returns:
        pq.ParquetDataset: The initialized Parquet dataset.
    """
    return pq.ParquetDataset(
        parquet_path, filters=partition_filters, filesystem=filesystem
    )


class SafeFragment:
    """
    Simple wrapper around `ParquetFileFragment` that allows to reinit the state of filesystem
    if aws session token has expired.
    """

    fragment: pa.dataset.ParquetFileFragment

    def __init__(self, fragment: pa.dataset.ParquetFileFragment):
        self.fragment = fragment

    def __repr__(self) -> str:
        out = ""
        out += "SafeFragment \n"
        out += "path = " + self.fragment.path + "\n"
        out += f"row_groups = {[int(rg.id) for rg in self.fragment.row_groups]} \n"
        out += f"physical_schema = \n {self.fragment.physical_schema} \n"
        return out

    def stable_hash(self, seed=None) -> int:
        return fragment_stable_hash(self.fragment, seed)

    @loading_retry
    def load(
        self,
        columns: Optional[List[str]] = None,
        use_threads: bool = False,
        add_fragment_traces: bool = True,
        add_partitioning_columns: bool = True,
    ) -> pa.Table:
        if columns is not None:
            fragment_columns = [
                col for col in columns if col in self.fragment.physical_schema.names
            ]
        else:
            fragment_columns = list(self.fragment.physical_schema.names)
        # adding technical columns for tracking
        if add_fragment_traces:
            fragment_columns = list(fragment_columns) + [
                "__batch_index",
                "__fragment_index",
                "__filename",
            ]
        try:
            fragment_table = self.fragment.to_table(
                columns=fragment_columns, use_threads=use_threads
            )

        except OSError as e:
            log.info(
                "could not load fragment, reinit the fragment state. Error: ", str(e)
            )
            self.fragment = loads(dumps(self.fragment))
            fragment_table = self.fragment.to_table(
                columns=fragment_columns, use_threads=use_threads
            )

        if add_partitioning_columns:
            fragment_table = add_partitioning_values(
                fragment_table, self.fragment, columns
            )
        if add_fragment_traces:
            fragment_table = add_fragments_trace(fragment_table, self.fragment)
        return fragment_table
