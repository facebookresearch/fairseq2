# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Dict, List, Optional, final

import pyarrow as pa
from pyarrow.dataset import get_partition_keys

from fairseq2.data import DataPipeline, DataPipelineBuilder
from fairseq2.data.parquet.arrow_transform.transform import (
    filter_list_with_min_max_length,
)
from fairseq2.data.parquet.fragment_loading import (
    FragmentLoadingConfig,
    NamedColumns,
    ParquetFragmentLoader,
)
from fairseq2.data.parquet.fragment_streaming import (
    FragmentStreamingConfig,
    ParquetFragmentStreamer,
)
from fairseq2.data.parquet.table_bucketing import TableBucketer, TableBucketingConfig
from fairseq2.data.parquet.utils import pyarrow_table_to_torch_dict
from fairseq2.data.text.tokenizers import TextTokenizer
from fairseq2.datasets import DataPipelineReader, DataReader
from fairseq2.datasets.parallel_text import (
    Direction,
    GenericParallelTextDataset,
    ParallelTextReadOptions,
)
from fairseq2.gang import Gang
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.nn.padding import pad_seqs
from fairseq2.typing import Device


@dataclass(kw_only=True)
class BiTextColumns(NamedColumns):
    source_text: str
    source_lang: str
    target_text: str
    target_lang: str
    domain: Optional[str] = None
    split: Optional[str] = "split"
    hard_negatives: Optional[str] = None
    quality_score: Optional[str] = None
    extra_columns: Optional[List[str]] = None


@dataclass(kw_only=True)
class ParquetParallelTextDatasetConfig:
    parquet_path: str
    columns: BiTextColumns
    filesystem: Optional[str] = None
    partition_filters: Optional[str] = None
    #
    nb_epochs: Optional[int] = None  # infinite if None
    cache: bool = False
    fragment_shuffle_window: int = -1  # do global shuffle for each direction
    direction_batch_size: int = 2

    sample_shuffle_window: int = 40_000
    max_tokens: int = 4_000
    min_seq_len: int = 0
    max_seq_len: int = 512
    order_by_length: bool = True

    direction_sample: bool = True
    direction_weights_manifest_path: Optional[str] = None

    nb_prefetch: int = 2
    num_parallel_calls: int = 4
    max_num_batches: Optional[int] = None
    seed: int = 123


class TokenizerMapper:
    def __init__(self, tokenizer: TextTokenizer) -> None:
        self._tokenizer = tokenizer

    @lru_cache(maxsize=10_000)
    def get_encoders(self, direction: Direction):
        source_mode = "source"
        if direction.origin != "primary":
            source_mode = f"{source_mode}_{direction.origin}"

        source_encoder = self._tokenizer.create_encoder(
            task="translation", lang=direction.source_lang, mode=source_mode
        )

        target_encoder = self._tokenizer.create_encoder(
            task="translation", lang=direction.target_lang, mode="target"
        )

        return source_encoder, target_encoder

    def __call__(self, table: pa.Table) -> pa.Table:
        direction = Direction(
            table["source_lang"][0].as_py(),
            table["target_lang"][0].as_py(),
            table["domain"][0].as_py() if "domain" in table.column_names else None,
        )

        source_encoder, target_encoder = self.get_encoders(direction)

        source_tokens = pa.array(
            table["source_text"]
            .to_pandas()
            .apply(lambda x: source_encoder(x).int().numpy()),
            type=pa.list_(pa.int64()),
        )
        target_tokens = pa.array(
            table["target_text"]
            .to_pandas()
            .apply(lambda x: target_encoder(x).int().numpy()),
            type=pa.list_(pa.int64()),
        )
        table = table.append_column("source_indices", source_tokens)
        table = table.append_column("target_indices", target_tokens)
        return table


@final
class ParquetParallelTextDataset:
    """Represents a parallel text dataset build on top of partitionned parquet dataset."""

    def __init__(self, name: str, config: ParquetParallelTextDatasetConfig) -> None:
        self._config = deepcopy(config)
        self.columns = self._config.columns
        self._name = name

        self.base_dataset_conifg = FragmentStreamingConfig(
            parquet_path=self._config.parquet_path,
            partition_filters=self._config.partition_filters,
            filesystem=self._config.filesystem,
            fragment_shuffle_window=self._config.fragment_shuffle_window,
            seed=self._config.seed,
            nb_epochs=self._config.nb_epochs,
        )

        self._direction_paths = self.get_all_direction()
        self._direction_weights = self.get_direction_weights(
            self._config.direction_weights_manifest_path
        )

    def reading_one_direction_pipeline(
        self, files, rank, world_size
    ) -> DataPipelineBuilder:
        dir_fragment_config = deepcopy(self.base_dataset_conifg)
        dir_fragment_config.parquet_path = files

        loading_config = FragmentLoadingConfig(
            columns=self._config.columns, cache=self._config.cache
        )
        bucketing_config = TableBucketingConfig(
            shuffle=True,
            batch_size=self._config.direction_batch_size,
            seed=2 * self._config.seed + 3 * rank,
        )

        # stream fragments for that given direction using sharding
        fragment_pipeline = ParquetFragmentStreamer(dir_fragment_config).build_pipeline(
            rank=rank, world_size=world_size
        )
        # now load the fragments
        fragment_pipeline = ParquetFragmentLoader(loading_config).build_pipeline(
            fragment_pipeline
        )
        # bucket and dispatch in short batches
        bucket_pipeline = TableBucketer(bucketing_config).build_pipeline(
            fragment_pipeline
        )

        return bucket_pipeline

    def get_all_direction(self) -> Dict[Direction, list[str]]:
        dataset = ParquetFragmentStreamer(self.base_dataset_conifg).dataset

        directions: Dict[Dict[Optional[str], [Direction, list[str]]]] = dict()
        for fragment in dataset._dataset.get_fragments(
            filter=dataset._filter_expression
        ):
            dd = get_partition_keys(fragment.partition_expression)
            split: str | None = dd.get(self.columns.split)
            if not split in directions:
                directions[split] = dict()

            source_lang = dd.get(self.columns.source_lang)
            target_lang = dd.get(self.columns.target_lang)
            origin = dd.get(self.columns.domain)
            direction = Direction(source_lang, target_lang, origin)
            if direction not in directions:
                directions[split][direction] = [fragment.path]
            else:
                directions[split][direction].append(fragment.path)

        return directions

    @staticmethod
    def get_direction_weights(
        manifest_path: str | None,
    ) -> Dict[Direction, float] | None:
        if manifest_path is None:
            return None

        weights = {}
        with Path(manifest_path).open("r") as f:
            for line in f:
                fields, weight = line.rstrip().split("\t")
                direction = GenericParallelTextDataset._parse_direction(fields)
                # XXX: map empty origin to primary
                direction.origin = direction.origin or "primary"
                weights[direction] = float(weight)

        return weights

    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        options: ParallelTextReadOptions = ParallelTextReadOptions(),
    ) -> DataReader[Seq2SeqBatch]:

        all_split_directions = self._direction_paths[split]

        # Determine the directions to read.
        direction = options.direction
        if direction is not None:
            if direction.origin is None:
                direction.origin = "primary"

        if direction is not None:
            if direction not in all_split_directions:
                raise ValueError(
                    f"`direction` must be a direction that exists in '{split}' split, but is '{direction}' instead. in {all_split_directions}"
                )
            all_split_directions = {direction: all_split_directions[direction]}

        if self._direction_weights is not None:
            weights, dir_files = zip(
                *[
                    (self._direction_weights[d], files_)
                    for d, files_ in all_split_directions.items()
                    if d in self._direction_weights
                ]
            )
        else:
            weights, dir_files = zip(
                *[(1.0, files_) for files_ in all_split_directions.values()]
            )

        if len(dir_files) == 1:  # short circuit for single direction
            builder = self.reading_one_direction_pipeline(
                dir_files[0], gang.rank, gang.size
            )
        else:
            reading_pipelines = []
            for files in dir_files:
                pipeline = self.reading_one_direction_pipeline(
                    files, gang.rank, gang.size
                )
                reading_pipelines.append(pipeline.and_return())

            if self._config.direction_sample:
                builder = DataPipeline.sample(
                    reading_pipelines,
                    weights=weights,
                    seed=self._config.seed + gang.rank,
                )
            else:
                builder = DataPipeline.concat(reading_pipelines)

        # tokenize
        builder = builder.map(TokenizerMapper(tokenizer=tokenizer))
        # filter out out-of-range examples
        builder = builder.map(
            partial(
                filter_list_with_min_max_length,
                columns=["source_indices", "target_indices"],
                min_length=self._config.min_seq_len,
                max_length=self._config.max_seq_len,
            ),
            num_parallel_calls=self._config.num_parallel_calls,
        )

        # this is maybe not needed (thanks to the bucketing)
        builder = builder.filter(lambda table: len(table) > 0)

        # bucket and dispatch in short batches
        bucketing_config = TableBucketingConfig(
            target_table_size=self._config.sample_shuffle_window,
            shuffle=True,
            total_batch_length=self._config.max_tokens,
            order_by_length=self._config.order_by_length,
            seed=2 * self._config.seed + gang.rank,  # any non-trival dependency
            length_columns=["source_indices", "target_indices"],
            nb_prefetch=self._config.nb_prefetch,
            length_reducer="max",
            cache=self._config.cache,
            num_parallel_calls=self._config.num_parallel_calls,
        )

        builder = TableBucketer(bucketing_config).build_pipeline(builder)

        if self._config.max_num_batches is not None:
            builder = builder.take(self._config.max_num_batches)

        # map to torch and pad
        to_final_batch = partial(
            self._to_batch,
            device=gang.device,
            padding_value=tokenizer.vocab_info.pad_idx,  # type: ignore
        )
        final_pipeline = builder.map(to_final_batch).and_return()

        return DataPipelineReader[Seq2SeqBatch](
            self._name, split, final_pipeline, gang, options=options, strict_state=False
        )

    @staticmethod
    def _to_batch(table: pa.Table, padding_value: int, device: Device) -> Seq2SeqBatch:
        torch_objects = pyarrow_table_to_torch_dict(
            table.select(["source_indices", "target_indices"])
        )
        source_seqs, source_padding_mask = pad_seqs(
            torch_objects["source_indices"], pad_value=padding_value, device=device  # type: ignore
        )
        target_seqs, target_padding_mask = pad_seqs(
            torch_objects["target_indices"], pad_value=padding_value, device=device  # type: ignore
        )

        return Seq2SeqBatch(
            source_seqs,
            source_padding_mask,
            target_seqs,
            target_padding_mask,
            table,
        )
