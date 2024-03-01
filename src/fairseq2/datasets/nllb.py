# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, final

from fairseq2.assets import AssetCard, AssetCardError
from fairseq2.assets import asset_store as default_asset_store
from fairseq2.assets import download_manager as default_download_manager
from fairseq2.data import (
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    create_bucket_sizes,
)
from fairseq2.data.text import TextTokenizer, read_text
from fairseq2.datasets.error import DatasetError
from fairseq2.datasets.loader import StandardDatasetLoader
from fairseq2.datasets.parallel_text_dataset import (
    AbstractParallelTextDataset,
    LangPair,
    load_parallel_text_dataset,
)
from fairseq2.gang import Gang
from fairseq2.typing import override

logger = logging.getLogger(__name__)

# TODO: FIX, INFER
num_parallel_calls = 10


@final
class NllbDataset(AbstractParallelTextDataset):
    """Represents an NLLB dataset."""

    _data_dir: Path
    _split_lang_pairs: Dict[str, Dict[LangPair, int]]

    def __init__(
        self,
        dataset_name: str,
        data_dir: Path,
        split_lang_pairs: Dict[str, Dict[LangPair, int]],
    ) -> None:
        """
        :param dataset_name:
            The name of the dataset.
        :param data_dir:
            The directory under which the language pair files reside.
        :param split_lang_pairs:
            The available language pairs per split.
        """
        super().__init__(dataset_name)

        self._data_dir = data_dir
        self._split_lang_pairs = split_lang_pairs

    @override
    def _build_pipeline(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        max_seq_len: int,
        max_num_tokens: int,
        bucket_by_length: bool,
        sample: bool,
        shuffle_window_size: int,
        num_prefetch: int,
        lang_pairs: Optional[Sequence[LangPair]],
    ) -> DataPipeline:
        splits = []

        for available_split in self._split_lang_pairs.keys():
            if available_split == split or available_split.startswith(split + "_"):
                splits.append(available_split)

        if not splits:
            raise ValueError(
                f"`split` must be a valid split name, but the {self._dataset_name} dataset has no split named '{split}'."
            )

        lang_pairs_to_read: Dict[LangPair, List[Tuple[str, int]]] = defaultdict(list)

        # Extract the language pairs along with their corpus sizes from the
        # requested split or splits.
        for split in splits:
            split_lang_pairs = self._split_lang_pairs[split]

            if lang_pairs is None:
                for lang_pair, size in split_lang_pairs.items():
                    lang_pairs_to_read[lang_pair].append((split, size))
            else:
                for lang_pair in lang_pairs:
                    try:
                        size = split_lang_pairs[lang_pair]
                    except KeyError:
                        continue

                    lang_pairs_to_read[lang_pair].append((split, size))

        # Ensure that we have found at least one split for each language pair.
        if lang_pairs is not None:
            for lang_pair in lang_pairs:
                if lang_pair not in lang_pairs_to_read:
                    raise ValueError(
                        f"All language pairs in `lang_pairs` must be available in the dataset, but '{lang_pair}' is not found in the '{split}' split of the {self.dataset_name} dataset."
                    )

        factory = _NllbDataPipelineFactory(
            self._dataset_name,
            self._data_dir,
            tokenizer,
            gang,
            lang_pairs_to_read,
            max_seq_len,
            max_num_tokens,
            bucket_by_length,
            sample,
            shuffle_window_size,
            num_prefetch,
        )

        return factory()

    @override
    def splits(self) -> List[str]:
        return list(self._split_lang_pairs.keys())

    @override
    def lang_pairs(self, split: str) -> List[LangPair]:
        try:
            return list(self._split_lang_pairs[split].keys())
        except KeyError:
            raise ValueError(
                f"`split` must be a valid split name, but the {self._dataset_name} dataset has no split named '{split}'."
            )


class _NllbDataPipelineFactory:
    dataset_name: str
    data_dir: Path
    tokenizer: TextTokenizer
    gang: Gang
    lang_pairs: Dict[LangPair, List[Tuple[str, int]]]
    max_seq_len: int
    max_num_tokens: int
    bucket_by_length: bool
    sample: bool
    shuffle_window_size: int
    num_prefetch: int

    def __init__(
        self,
        dataset_name: str,
        data_dir: Path,
        tokenizer: TextTokenizer,
        gang: Gang,
        lang_pairs: Dict[LangPair, List[Tuple[str, int]]],
        max_seq_len: int,
        max_num_tokens: int,
        bucket_by_length: bool,
        sample: bool,
        shuffle_window_size: int,
        num_prefetch: int,
    ) -> None:
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.gang = gang
        self.lang_pairs = lang_pairs
        self.max_seq_len = max_seq_len
        self.max_num_tokens = max_num_tokens
        self.bucket_by_length = bucket_by_length
        self.sample = sample
        self.shuffle_window_size = shuffle_window_size
        self.num_prefetch = num_prefetch

    def __call__(self) -> DataPipeline:
        lang_pair_pipelines: List[DataPipeline] = []

        # The number of examples to be read per language pair.
        lang_pair_sizes: List[int] = []

        # The total number of examples to be read.
        total_size = 0

        for lang_pair, splits in self.lang_pairs.items():
            for split, size in splits:
                # Sharding always returns a multiple of `gang.size` examples
                # from the dataset and drops any remainder if the dataset is
                # not evenly distributed among processes. Account for it.
                size -= size % self.gang.size

                if size == 0:
                    raise DatasetError(
                        f"The '{split}' split of the {self.dataset_name} dataset has no examples for the language pair '{lang_pair}'."
                    )

                pipeline = self._create_lang_pair_pipeline(split, lang_pair)

                lang_pair_pipelines.append(pipeline)

                lang_pair_sizes.append(size)

                total_size += size

        if self.sample:
            # Sample the language pairs in proportion to their corpus size.
            pipeline_bld = DataPipeline.sample(
                lang_pair_pipelines, weights=[s / total_size for s in lang_pair_sizes]
            )
        else:
            pipeline_bld = DataPipeline.concat(lang_pair_pipelines)

        return self._build_pipeline(pipeline_bld)

    def _create_lang_pair_pipeline(
        self, split: str, lang_pair: LangPair
    ) -> DataPipeline:
        source_lang, target_lang = lang_pair

        source_file, target_file = self._get_lang_pair_files(
            split, source_lang, target_lang
        )

        source_pipeline_bld = read_text(source_file, rtrim=True, memory_map=True)
        target_pipeline_bld = read_text(target_file, rtrim=True, memory_map=True)

        if self.gang.size > 1:
            source_pipeline_bld.shard(self.gang.rank, self.gang.size)
            target_pipeline_bld.shard(self.gang.rank, self.gang.size)

        # Initialize the token encoders for the source and target languages.
        source_encoder_mode = "source"

        # Check if we have a train split with a specific data source.
        if split.startswith("train_"):
            source_encoder_mode = f"{source_encoder_mode}_{split[6:]}"

        source_encoder = self.tokenizer.create_encoder(
            task="translation", lang=source_lang, mode=source_encoder_mode
        )

        target_encoder = self.tokenizer.create_encoder(
            task="translation", lang=target_lang, mode="target"
        )

        source_pipeline_bld.map(source_encoder, num_parallel_calls=num_parallel_calls)
        target_pipeline_bld.map(target_encoder, num_parallel_calls=num_parallel_calls)

        source_pipeline = source_pipeline_bld.and_return()
        target_pipeline = target_pipeline_bld.and_return()

        # Include the language pair name and the line number with each example
        # for troubleshooting.
        split_ = DataPipeline.constant(split).and_return()

        lang_pair_ = DataPipeline.constant(lang_pair).and_return()

        line_nr = DataPipeline.count(
            start=self.gang.rank, step=self.gang.size
        ).and_return()

        # Zip the source and target pipelines along with the pseudo pipelines
        # into one.
        names = ["split", "lang_pair", "line_nr", "source_indices", "target_indices"]

        pipeline_bld = DataPipeline.zip(
            [split_, lang_pair_, line_nr, source_pipeline, target_pipeline], names
        )

        return pipeline_bld.and_return()

    def _get_lang_pair_files(
        self, split: str, source_lang: str, target_lang: str
    ) -> Tuple[Path, Path]:
        source_filename = f"{split}.{source_lang}-{target_lang}.{source_lang}"
        target_filename = f"{split}.{source_lang}-{target_lang}.{target_lang}"

        source_file = self.data_dir.joinpath(source_filename)
        target_file = self.data_dir.joinpath(target_filename)

        if not source_file.exists():
            raise DatasetError(
                f"The source language file '{source_file}' is not found in the {self.dataset_name} dataset."
            )

        if not target_file.exists():
            raise DatasetError(
                f"The target language file '{target_file}' is not found in the {self.dataset_name} dataset."
            )

        return source_file, target_file

    def _build_pipeline(self, pipeline_bld: DataPipelineBuilder) -> DataPipeline:
        # Shuffle examples.
        if self.shuffle_window_size > 0:
            pipeline_bld.shuffle(self.shuffle_window_size)

        if self.bucket_by_length:
            # Bucket by the length of the source or target sequence. The longer
            # one will be considered the length of the example.
            bucket_sizes = create_bucket_sizes(
                self.max_num_tokens, self.max_seq_len, min_seq_len=4
            )

            pipeline_bld.bucket_by_length(
                bucket_sizes,
                selector="source_indices,target_indices",
                skip_long_examples=True,
            )
        else:
            # TODO(balioglu): FIX!
            pipeline_bld.bucket(32)

        # Collate bucketed examples into a batch.
        collater = Collater(pad_value=self.tokenizer.vocab_info.pad_idx)

        pipeline_bld.map(collater, num_parallel_calls=num_parallel_calls)

        # Prefetch examples in a background thread.
        if self.num_prefetch > 0:
            pipeline_bld.prefetch(self.num_prefetch)

        return pipeline_bld.and_return()


def _create_nllb_dataset(path: Path, card: AssetCard) -> NllbDataset:
    split_lang_pairs: Dict[str, Dict[LangPair, int]] = {}

    splits_field = card.field("splits")

    for split in splits_field.as_dict(dict).keys():
        lang_pairs = {}

        for key, size in splits_field.field(split).as_dict(int).items():
            try:
                source_lang, target_lang = key.split("-")
            except ValueError as ex:
                raise AssetCardError(
                    f"The items of the field 'splits.{split}' of the asset card '{card.name}' must represent language pairs, but '{key}' does not represent a language pair."
                ) from ex

            lang_pair = LangPair(source_lang, target_lang)

            lang_pairs[lang_pair] = size

        split_lang_pairs[split] = lang_pairs

    return NllbDataset(card.name, path, split_lang_pairs)


load_nllb_dataset = StandardDatasetLoader[NllbDataset](
    default_asset_store, default_download_manager, _create_nllb_dataset
)

load_parallel_text_dataset.register_loader("nllb", load_nllb_dataset)
