# Coeyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import concurrent.futures
from dataclasses import dataclass
from functools import reduce
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc  # noqa: F401
import pyarrow.parquet as pq
import torch
from pyarrow.dataset import get_partition_keys

from fairseq2.data.data_pipeline import (
    DataPipelineBuilder,
    read_iterator,
    read_sequence,
)
from fairseq2.data.parquet.fragment_streaming.config import ParquetDatasetLimitOptions
from fairseq2.data.parquet.utils import (
    circular_shift_left,
    get_dataset_fragments,
    split_fragment_in_row_groups,
)
from fairseq2.logging import log

try:
    from itertools import batched  # type: ignore
except ImportError:
    from itertools import islice

    def batched(iterable, n):
        # batched('ABCDEFG', 3) --> ABC DEF G
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch


def process_filter(
    filters: Optional[str | list[str] | pa.compute.Expression],
) -> Optional[pa.compute.Expression]:
    """Process the filters to be applied to the dataset.
    - python string is evaluated to get the expression using `eval` (in particular, symbols `pa`, `pc`, and `pq` can be used)
       e.g. 'pc.is_in(pc.field("lang"), pa.array(["en", "fr"]))'

    - list of filters is reduced to a single expression using `&` operator

    """
    if filters is None or isinstance(filters, pa.compute.Expression):
        return filters

    if isinstance(filters, str):
        return pq.filters_to_expression(eval(filters))

    if isinstance(filters, list):
        list_of_filter = [process_filter(f) for f in filters]
        return reduce(lambda x, y: x & y, list_of_filter)

    raise ValueError(f"Unknown type of filters: {type(filters)}")


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


def _parquet_fragments_to_pipeline_builder(
    file_ds_fragments: List[pa.dataset.Fragment],
    nb_epochs: Optional[int] = 1,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> DataPipelineBuilder:
    if nb_epochs is None:
        # schedule all fragments in 100 epochs and next repeat them
        # just to avoid frequent reset pipeline
        _nb_epochs = 100
    else:
        _nb_epochs = nb_epochs

    if shuffle:
        if seed is None:
            seed = int(torch.randint(0, 2**31, ()).item())

        rsg = np.random.RandomState(seed)
        ds_fragments_ = np.asarray(file_ds_fragments, dtype="O")
        ds_fragments = np.concatenate(
            [rsg.permutation(ds_fragments_) for _ in range(_nb_epochs)]
        ).tolist()
    else:
        ds_fragments = file_ds_fragments * _nb_epochs

    pipeline_builder = read_sequence(ds_fragments)

    if nb_epochs is None:
        pipeline_builder = pipeline_builder.repeat(None)

    return pipeline_builder


def list_parquet_fragments(
    parquet_ds: pq.ParquetDataset,
    nb_epochs: Optional[int] = 1,
    split_to_row_groups: bool = True,
    shuffle: bool = True,
    seed: Optional[int] = None,
    limit_options: Optional[ParquetDatasetLimitOptions] = None,
    nb_jobs: int = 10,
) -> DataPipelineBuilder:

    if limit_options is None:
        limit_options = ParquetDatasetLimitOptions()

    file_ds_fragments = get_dataset_fragments(parquet_ds, parquet_ds._filter_expression)
    proxy_ds_path = "/".join(parquet_ds.files[0].split("=")[0].split("/")[:-1])

    if limit_options.fraction_of_files is not None:
        file_ds_fragments = file_ds_fragments[
            : max(
                int(round(limit_options.fraction_of_files * len(file_ds_fragments))), 1
            )
        ]
        log.info(
            f"{proxy_ds_path} : reducing number of files to {len(file_ds_fragments)} because of fraction_of_files={limit_options.fraction_of_files}"
        )
    if limit_options.nb_files is not None and limit_options.nb_files < len(
        file_ds_fragments
    ):
        file_ds_fragments = file_ds_fragments[: limit_options.nb_files]
        log.info(
            f"{proxy_ds_path} : reducing number of files to {len(file_ds_fragments)} because of nb_files={limit_options.nb_files}"
        )

    output_fragments = []
    total_nb_rows: int = 0
    if split_to_row_groups:
        total_nb_fragments = 0
        early_stop = False
        with concurrent.futures.ThreadPoolExecutor(max_workers=nb_jobs) as executor:
            for batch_of_files in batched(file_ds_fragments, 20 * nb_jobs):
                # Submit tasks to the executor for splitting fragments
                futures = [
                    executor.submit(split_fragment_in_row_groups, ff)
                    for ff in batch_of_files
                ]
                row_groups = [future.result() for future in futures]
                new_file_fragments = [x for y in row_groups for x in y]
                new_file_fragments_stats: List[int]
                if limit_options.nb_rows is not None:
                    # Submit tasks to the executor for calculating stats
                    futures = [
                        executor.submit(
                            lambda frag: int(frag.row_groups[0].num_rows), ff
                        )
                        for ff in new_file_fragments
                    ]
                    new_file_fragments_stats = [future.result() for future in futures]
                else:
                    new_file_fragments_stats = [0] * len(new_file_fragments)

                for nb_row, frag in zip(new_file_fragments_stats, new_file_fragments):
                    output_fragments.append(frag)
                    total_nb_rows += nb_row
                    total_nb_fragments += 1
                    if (
                        limit_options.nb_fragments is not None
                        and total_nb_fragments >= limit_options.nb_fragments
                    ):
                        early_stop = True
                        if limit_options.nb_rows is not None:
                            log.info(
                                f"{proxy_ds_path} : nb_fragments limit {limit_options.nb_fragments} was reached with around {total_nb_rows} rows"
                            )
                        else:
                            log.info(
                                f"{proxy_ds_path} : nb_fragments limit {limit_options.nb_fragments} was reached"
                            )
                        break
                    if (
                        limit_options.nb_rows is not None
                        and total_nb_rows >= limit_options.nb_rows
                    ):
                        early_stop = True
                        log.info(
                            f"{proxy_ds_path} : nb_rows limit {limit_options.nb_rows} was reached with around {total_nb_fragments} fragments"
                        )
                        break
                if early_stop:
                    break
    else:
        for frag in file_ds_fragments[: limit_options.nb_fragments]:
            output_fragments.append(frag)
            if limit_options.nb_rows is not None:
                total_nb_rows += frag.count_rows()
                if total_nb_rows >= limit_options.nb_rows:
                    break

    return _parquet_fragments_to_pipeline_builder(
        output_fragments,
        nb_epochs=nb_epochs,
        shuffle=shuffle,
        seed=seed,
    )


@dataclass
class PFSState:
    nb_fully_read_files: int = 0
    nb_current_file_read_fragements: int = 0


class ParquetFragmentStreamer:
    def __init__(
        self,
        parquet_ds: pq.ParquetDataset,
        split_to_row_groups: bool = True,
        limit_options: Optional[ParquetDatasetLimitOptions] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        relative_files_circular_shift: float = 0.0,
        read_state: Optional[PFSState] = None,
    ):
        self.parquet_ds = parquet_ds
        self.split_to_row_groups = split_to_row_groups
        self.limit_options = limit_options or ParquetDatasetLimitOptions()
        self.shuffle = shuffle
        self.seed = seed
        self.relative_files_circular_shift = relative_files_circular_shift

        if not (0.0 <= self.relative_files_circular_shift <= 1.0):
            raise ValueError(
                f"relative_files_circular_shift must be between 0.0 and 1.0, got {self.relative_files_circular_shift} instead"  # fmt: skip
            )

        if read_state is not None:
            self.state = read_state
        else:
            self.reset_state()

    def reset_state(self, seed: Optional[int] = None):
        self.state = PFSState()
        if seed is not None:
            self.seed = seed

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.parquet_ds,
                self.split_to_row_groups,
                self.limit_options,
                self.shuffle,
                self.seed,
                self.relative_files_circular_shift,
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

    def __iter__(self):
        limit_options = self.limit_options

        file_ds_fragments = self.truncate_files(
            self.parquet_ds,
            limit_options.fraction_of_files,
            limit_options.nb_files,
        )
        if self.shuffle:
            random_state = np.random.RandomState(self.seed)
            np_file_ds_fragments = np.array(file_ds_fragments, dtype="O")
            random_state.shuffle(np_file_ds_fragments)
            file_ds_fragments = np_file_ds_fragments.tolist()

        # circular shift files by a give ratio
        if (
            self.relative_files_circular_shift > 0.0
            and self.relative_files_circular_shift < 1.0
        ):
            file_ds_fragments = circular_shift_left(
                file_ds_fragments,
                int(self.relative_files_circular_shift * len(file_ds_fragments)),
            )

        if limit_options.nb_fragments is not None or limit_options.nb_rows is not None:
            raise NotImplementedError(
                "`nb_fragments` and `nb_rows` are not supported for StreamingParquetDataset"
                "use `list_parquet_fragments` instead"
            )
        # TODO: shuffle file_ds_fragments differently for each rank/world size

        if not self.split_to_row_groups:
            for frag in file_ds_fragments[self.state.nb_fully_read_files :]:
                self.state.nb_fully_read_files += 1
                yield frag
        else:

            for new_file in file_ds_fragments[self.state.nb_fully_read_files :]:
                new_file_fragments = split_fragment_in_row_groups(new_file)
                new_file_fragments = new_file_fragments[
                    self.state.nb_current_file_read_fragements :
                ]

                for frag in new_file_fragments:
                    # increate before yield
                    self.state.nb_current_file_read_fragements += 1
                    yield frag

                # only when full file is read we increament this
                self.state.nb_fully_read_files += 1
                self.state.nb_current_file_read_fragements = 0


@dataclass
class ShuffledIteratorState:
    epoch_count: int
    current_window: List[Any]
    index: int
    random_state: np.random.RandomState


class ShuffledIterator(Iterator[Any]):
    def __init__(
        self,
        base_iterator,
        window_size: int,
        nb_epoch: Optional[int],
        seed: Optional[int],
        state: Optional[ShuffledIteratorState] = None,
    ):
        self.base_iterator = base_iterator
        self.window_size = window_size
        self.seed = seed
        self.nb_epoch = nb_epoch

        if state is None:
            state = ShuffledIteratorState(
                random_state=np.random.RandomState(self.seed),
                epoch_count=0,
                current_window=[],
                index=0,
            )
        self.state = state
        self.window_iterator = None

    def reset_state(self):
        self.state.random_state = np.random.RandomState(self.seed)
        self.state.epoch_count = 0
        self._reset_inner(self.seed)

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.base_iterator,
                self.window_size,
                self.nb_epoch,
                self.seed,
                self.state,
            ),
        )

    def _reset_inner(self, seed: Optional[int]):
        self.base_iterator.reset_state(seed)
        self.state.index = 0
        self.state.current_window = []
        self.window_iterator = None

    def __iter__(self):
        return self

    def repeat_iterator(self):
        while (self.nb_epoch is None) or (self.state.epoch_count < self.nb_epoch):
            for element in self.base_iterator:
                yield element

            self.state.epoch_count += 1

            if self.seed is not None:
                seed = self.seed + self.state.epoch_count
            else:
                seed = None
            self._reset_inner(seed)

    def __next__(self) -> Any:
        if self.window_iterator is None:
            self.window_iterator = batched(self.repeat_iterator(), self.window_size)  # type: ignore

        assert self.window_iterator is not None

        if self.state.index >= len(self.state.current_window):
            window = next(self.window_iterator)
            window = np.array(window, dtype="O")
            self.state.random_state.shuffle(window)
            self.state.current_window = window
            self.state.index = 0

        # Return the next element from the current window
        result = self.state.current_window[self.state.index]
        self.state.index += 1
        return result


def stream_parquet_fragments(
    parquet_ds: pq.ParquetDataset,
    nb_epochs: Optional[int],
    split_to_row_groups: bool = True,
    shuffle: bool = True,
    seed: Optional[int] = None,
    limit_options: Optional[ParquetDatasetLimitOptions] = None,
    shuffling_window: int = 200,
    files_circular_shift: float = 0.0,
) -> DataPipelineBuilder:

    fragments_iterator = ParquetFragmentStreamer(
        parquet_ds=parquet_ds,
        split_to_row_groups=split_to_row_groups,
        limit_options=limit_options,
        shuffle=shuffle,
        seed=seed,
        relative_files_circular_shift=files_circular_shift,
    )

    def reset_fn(iterator):
        iterator.reset_state()
        return iterator

    pipeline = read_iterator(
        ShuffledIterator(
            fragments_iterator,
            window_size=shuffling_window if shuffle else 1,
            nb_epoch=nb_epochs,
            seed=seed,
        ),
        reset_fn,
        infinite=False,
    )

    return pipeline


class RejectionDistributionSmoother:
    def __init__(
        self,
        partition_groups: List[str],
        alpha: float = 0.5,
        min_count: int = 10,
        seed: int | None = 0,
    ):
        """
        RejectionDistributionSmoother is used to balance fragments the distribution of samples across partition groups.

        This class implements a rejection sampling technique to smooth the distribution of data
        across different partition groups in a Parquet dataset. It keeps track of the frequency
        of each partition group and uses these frequencies to decide whether to accept or reject
        a fragment based on its partition values.

        Args:
            partition_groups (List[str]): List of partition column names to group by.
            alpha (float, optional): Controls the smoothing effect.
                                    0 = original distribution,
                                    1 = uniform distribution.
                                    Defaults to 0.5.
            min_count (int, optional): Minimum count to use for frequency normalization. Defaults to 10.
            seed (int | None, optional): Random seed for reproducibility. Defaults to 0.

        Example:
            ```python

            # Create a smoother for balancing across 'lang' and 'domain' partitions
            smoother = RejectionDistributionSmoother(
                partition_groups=["lang", "domain"],
                alpha=0.3,  # Partial smoothing between original and uniform
                min_count=20,
                seed=42
            )

            # Create a pipeline that streams fragments and applies the smoother

            fragment_stream_config=FragmentStreamingConfig(
                parquet_path="path/to/partitioned_parquet",
                partition_filters='pc.field("split") == "train"',
                nb_epochs=None,
                fragment_shuffle_window=100)

            build_pipeline = ParquetFragmentStreamer(
                config=config.fragment_stream_config
            ).build_pipeline(rank=0, world_size=1)

            # Note that parquet dataset must be partitioned on the partition_groups ("lang" and "domain" here)!!
            balanced_build_pipeline = build_pipeline.map(smoother).filter(lambda x: x is not None)
            ```
        """

        self.partition_groups = partition_groups
        self.freqs: Dict[Any, int] = {}
        self.alpha = alpha
        self.min_count = min_count
        self.random_state = np.random.RandomState(seed)

    def extract(self, fragment: pa.dataset.ParquetFileFragment):
        partition_dict = get_partition_keys(fragment.partition_expression)
        sorted_values = tuple(
            [partition_dict.get(key) for key in self.partition_groups]
        )
        cnt = sum(rr.num_rows for rr in fragment.row_groups)
        return sorted_values, cnt

    def __call__(self, element: pa.dataset.ParquetFileFragment):
        extracted_element, cnt = self.extract(element)
        self.freqs[extracted_element] = self.freqs.get(extracted_element, 0) + cnt
        normalization_weight = sum(
            1 / max(self.freqs[elem], self.min_count) for elem in self.freqs
        )
        acceptance_prob = 1.0 / (self.freqs[extracted_element] * normalization_weight)

        if self.random_state.rand() < acceptance_prob**self.alpha:
            return element
        else:
            return None
