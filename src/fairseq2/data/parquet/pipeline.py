# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pickle import dumps, loads
from typing import Any, Iterator, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from retrying import retry

from fairseq2.data.data_pipeline import DataPipeline, DataPipelineBuilder, read_sequence
from fairseq2.data.parquet.configs import ParquetDatasetLimitOptions
from fairseq2.data.parquet.transform import add_fragments_trace
from fairseq2.data.parquet.utils import (
    add_partitioning_values,
    compute_length_splits,
    compute_rows_length,
    get_dataset_fragments,
    split_fragment_in_row_groups,
    table_func_wrap,
)
from fairseq2.logging import log

# --- tested above --- #


loading_retry = retry(
    retry_on_exception=lambda exception: isinstance(exception, OSError),
    stop_max_attempt_number=1,
    wait_exponential_multiplier=2,
    wait_exponential_max=20,
)


def init_parquet_dataset(
    parquet_path: str,
    filters: Optional[pa.dataset.Expression] = None,
) -> pq.ParquetDataset:
    """
    Initialize a Parquet dataset.
    Leaving `filesystem` to None will trigger the detection of the filesystem.

    Args:
        parquet_path (str): The path to the Parquet dataset.
        filters (Optional[pa.dataset.Expression]): Filters to apply to the dataset.

    Returns:
        pq.ParquetDataset: The initialized Parquet dataset.
    """
    return pq.ParquetDataset(parquet_path, filters=filters, filesystem=None)


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

    @loading_retry
    def load(self, columns: Optional[List[str]] = None) -> pa.Table:
        if columns is not None:
            fragment_columns = [
                col for col in columns if col in self.fragment.physical_schema.names
            ]
        else:
            fragment_columns = self.fragment.physical_schema.names
        # adding technical columns for tracking
        fragment_columns = list(fragment_columns) + [
            "__batch_index",
            "__fragment_index",
            "__filename",
        ]
        try:
            fragment_table = self.fragment.to_table(
                columns=fragment_columns, use_threads=False
            )

        except OSError as e:
            # XXX: this will reinit default aws creds if they were not provided explicitly
            # tested only on aws cluster only !
            log.info(
                "could not load fragment, reinit the fragment state. Error: ", str(e)
            )
            self.fragment = loads(dumps(self.fragment))
            fragment_table = self.fragment.to_table(
                columns=fragment_columns, use_threads=False
            )

        fragment_table = add_partitioning_values(fragment_table, self.fragment, columns)
        fragment_table = add_fragments_trace(fragment_table, self.fragment)
        return fragment_table


@dataclass
class PFSState:
    nb_fully_read_files: int = 0
    nb_current_file_read_fragements: int = 0
    total_nb_fragments: int = 0
    total_nb_rows: int = 0


class ParquetFragmentStreamer:
    def __init__(
        self,
        parquet_ds: pq.ParquetDataset,
        split_to_row_groups: bool = True,
        limit_options: Optional[ParquetDatasetLimitOptions] = None,
        read_state: Optional[PFSState] = None,
    ):
        self.split_to_row_groups = split_to_row_groups
        self.limit_options = limit_options or ParquetDatasetLimitOptions()
        self.parquet_ds = parquet_ds

        if read_state is not None:
            self.state = read_state
        else:
            self.reset_state()

    def reset_state(self) -> None:
        self.state = PFSState()

    def __reduce__(self) -> tuple[type[ParquetFragmentStreamer], tuple[Any, ...]]:
        return (
            self.__class__,
            (
                self.parquet_ds,
                self.split_to_row_groups,
                self.limit_options,
                self.state,
            ),
        )

    def truncate_files(
        self,
        parquet_ds: pq.ParquetDataset,
        fraction_of_files: Optional[float],
        nb_files: Optional[int],
    ) -> List[pa.dataset.Fragment]:
        file_ds_fragments = get_dataset_fragments(
            parquet_ds, parquet_ds._filter_expression
        )
        self.proxy_ds_path = "/".join(parquet_ds.files[0].split("=")[0].split("/")[:-1])
        log.info(
            f"{self.proxy_ds_path} : full number of files {len(file_ds_fragments)}"
        )

        if fraction_of_files is not None:
            file_ds_fragments = file_ds_fragments[
                : max(
                    int(round(fraction_of_files * len(file_ds_fragments))),
                    1,
                )
            ]
            log.info(
                f"{self.proxy_ds_path} : reducing number of files to {len(file_ds_fragments)} because of fraction_of_files={fraction_of_files}"
            )
        if nb_files is not None and nb_files < len(file_ds_fragments):
            file_ds_fragments = file_ds_fragments[:nb_files]
            log.info(
                f"{self.proxy_ds_path} : reducing number of files to {len(file_ds_fragments)} because of nb_files={nb_files}"
            )
        return file_ds_fragments

    def __iter__(self) -> Iterator[pa.dataset.Fragment]:
        limit_options = self.limit_options

        file_ds_fragments = self.truncate_files(
            self.parquet_ds,
            limit_options.fraction_of_files,
            limit_options.nb_files,
        )

        if not self.split_to_row_groups:
            for frag in file_ds_fragments[
                self.state.nb_fully_read_files : limit_options.nb_fragments  # noqa: E203
            ]:
                self.state.nb_fully_read_files += 1
                yield frag

                if limit_options.nb_rows is not None:
                    self.state.total_nb_rows += frag.count_rows()
                    if self.state.total_nb_rows >= limit_options.nb_rows:
                        break
        else:
            early_stop = False
            log.info(f"{self.proxy_ds_path} : starting split in row groups")

            for new_file in file_ds_fragments[
                self.state.nb_fully_read_files :  # noqa: E203
            ]:
                new_file_fragments = split_fragment_in_row_groups(new_file)
                new_file_fragments = new_file_fragments[
                    self.state.nb_current_file_read_fragements :  # noqa: E203
                ]
                if limit_options.nb_rows is not None:
                    new_file_fragments_stats = [
                        frag.row_groups[0].num_rows for frag in new_file_fragments
                    ]
                else:
                    new_file_fragments_stats = [0] * len(new_file_fragments)

                for nb_row, frag in zip(new_file_fragments_stats, new_file_fragments):
                    self.state.total_nb_rows += nb_row
                    self.state.total_nb_fragments += 1
                    self.state.nb_current_file_read_fragements += (
                        1  # increate before yield
                    )
                    yield frag

                    if (
                        limit_options.nb_fragments is not None
                        and self.state.total_nb_fragments >= limit_options.nb_fragments
                    ):
                        early_stop = True
                        if limit_options.nb_rows is not None:
                            log.info(
                                f"{self.proxy_ds_path} : nb_fragments limit {limit_options.nb_fragments} was reached with around {self.state.total_nb_rows} rows"
                            )
                        else:
                            log.info(
                                f"{self.proxy_ds_path} : nb_fragments limit {limit_options.nb_fragments} was reached"
                            )
                        break
                    if (
                        limit_options.nb_rows is not None
                        and self.state.total_nb_rows >= limit_options.nb_rows
                    ):
                        early_stop = True
                        log.info(
                            f"{self.proxy_ds_path} : nb_rows limit {limit_options.nb_rows} was reached with around {self.state.total_nb_fragments} fragments"
                        )
                        break
                if early_stop:
                    break
                # only when full file is read we increament this
                self.state.nb_fully_read_files += 1
                self.state.nb_current_file_read_fragements = 0


def parquet_fragments_to_pipeline_builder(
    file_ds_fragments: List[pa.dataset.Fragment],
    nb_epochs: int = 1,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> DataPipelineBuilder:

    if shuffle:
        if seed is None:
            seed = int(torch.randint(0, 2**31, ()).item())

        rsg = np.random.RandomState(seed)
        ds_fragments_ = np.asarray(file_ds_fragments, dtype="O")
        ds_fragments = np.concatenate(
            [rsg.permutation(ds_fragments_) for _ in range(nb_epochs)]
        ).tolist()
    else:
        ds_fragments = file_ds_fragments * nb_epochs

    pipeline_builder = read_sequence(ds_fragments)
    pipeline_builder = pipeline_builder.map(SafeFragment)
    return pipeline_builder


def list_parquet_fragments(
    parquet_path: str,
    filters: Optional[pa.dataset.Expression] = None,
    columns: Optional[List[str]] = None,
    split_to_row_groups: bool = True,
    shuffle_window: Optional[int] = None,
    seed: int = 2,
) -> DataPipelineBuilder:
    dataset = init_parquet_dataset(parquet_path, filters=filters)
    columns = columns or dataset.schema.names
    if not set(columns).issubset(set(dataset.schema.names)):
        raise ValueError(
            f"columns {sorted(set(columns) - set(dataset.schema.names))} are not found in the dataset schema"
        )

    pipeline_builder = read_sequence(get_dataset_fragments(dataset, filters))

    if shuffle_window is not None:
        # shuffle them in full memory since fragments are already known
        pipeline_builder = pipeline_builder.shuffle(shuffle_window=0, seed=seed)

    if split_to_row_groups:
        pipeline_builder = pipeline_builder.yield_from(
            lambda fragment: read_sequence(
                split_fragment_in_row_groups(fragment)
            ).and_return()
        )
        if shuffle_window is not None:
            pipeline_builder = pipeline_builder.shuffle(
                shuffle_window=shuffle_window, seed=seed + 1
            )

    return pipeline_builder


def build_iterator_over_one_table(
    table: pa.Table,
    order_by_length: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_tokens: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
    num_parallel_calls: int = 8,
) -> DataPipeline:
    random_state = np.random.RandomState(seed)
    if order_by_length is not None:
        length_col = compute_rows_length(table[order_by_length])
        # add small perturbation to avoid same sample appear together during different epochs
        if shuffle:
            perturbation = random_state.randint(
                0,
                np.quantile(length_col, 0.001).astype(np.int32) + 2,
                len(length_col),
            )
            length_col += np.asarray(perturbation, dtype=np.int32)
    else:
        if shuffle:
            length_col = random_state.randint(0, 2**23, len(table))
        else:
            length_col = np.zeros(len(table), dtype=np.int32)

    if batch_size is not None:
        order_tt = pa.Table.from_arrays(
            [pa.array(np.argsort(length_col, kind="stable"))], ["order"]
        )
        batches = [ind["order"] for ind in order_tt.to_batches(batch_size)]
    elif max_tokens is not None:
        batches = compute_length_splits(length_col, max_tokens)
    else:
        raise ValueError("unknown batching method")

    if shuffle:
        batches = [batches[i] for i in random_state.permutation(len(batches))]

    return (
        read_sequence(batches)
        .map(
            table_func_wrap(lambda ind: table.take(ind).combine_chunks()),
            num_parallel_calls=num_parallel_calls,
        )
        .and_return(max_num_warnings=4)
    )
