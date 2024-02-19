# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict, cast, final

from torch import Tensor

from fairseq2.assets import AssetCard, AssetCardError
from fairseq2.assets import asset_store as default_asset_store
from fairseq2.assets import download_manager as default_download_manager
from fairseq2.data import (
    Collater,
    CString,
    DataPipeline,
    DataPipelineBuilder,
    SequenceData,
    create_bucket_sizes,
)
from fairseq2.data.text import TextTokenizer, read_text
from fairseq2.datasets import (
    DatasetError,
    LangPair,
    MultilingualTextDataset,
    StandardDatasetLoader,
    load_multilingual_text_dataset,
)
from fairseq2.gang import Gang
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.typing import Device, finaloverride

logger = logging.getLogger(__name__)

# TODO: FIX, INFER
num_parallel_calls = 10


@final
class NllbDataset(MultilingualTextDataset):
    """Represents a multilingual NLLB text dataset."""

    dataset_name: str
    data_dir: Path
    splits: Dict[str, Dict[LangPair, int]]

    def __init__(
        self,
        dataset_name: str,
        data_dir: Path,
        splits: Dict[str, Dict[LangPair, int]],
    ) -> None:
        """
        :param dataset_name:
            The name of the dataset.
        :param data_dir:
            The directory under which the language pair files reside.
        :param splits:
            The available splits.
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.splits = splits

    @finaloverride
    def read(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        *,
        max_seq_len: int = 512,
        max_num_tokens: int = 2048,
        shuffle_window_size: int = 10_000,
        eval_batch_size: int = 32,
        num_prefetch: int = 10,
        lang_pairs: Optional[Sequence[LangPair]] = None,
    ) -> DataPipeline:
        split_names = []

        for split_name in self.splits.keys():
            if split_name == split or split_name.startswith(split + "_"):
                split_names.append(split_name)

        if not split_names:
            raise ValueError(
                f"`split` must be a valid split name, but the {self.dataset_name} dataset has no split named '{split}'."
            )

        splits = {}

        # Extract the language pairs along with their sizes from the requested
        # split or splits.
        for split_name in split_names:
            split_lang_pairs = self.splits[split_name]

            if lang_pairs is None:
                lang_pairs_to_read = split_lang_pairs
            else:
                lang_pairs_to_read = {}

                for lang_pair in lang_pairs:
                    try:
                        lang_pairs_to_read[lang_pair] = split_lang_pairs[lang_pair]
                    except KeyError:
                        raise ValueError(
                            f"All language pairs in `lang_pairs` must be available in the dataset, but '{lang_pair}' is not found in the '{split}' split of the {self.dataset_name} dataset."
                        )

            splits[split_name] = lang_pairs_to_read

        train = split == "train" or split.startswith("train_")

        factory = _NllbDataPipelineFactory(
            self.dataset_name,
            self.data_dir,
            tokenizer,
            splits,
            gang,
            train,
            max_seq_len,
            max_num_tokens,
            shuffle_window_size,
            eval_batch_size,
            num_prefetch,
        )

        return factory()

    @finaloverride
    def get_splits(self) -> List[str]:
        return list(self.splits.keys())

    @finaloverride
    def get_lang_pairs(self, split: str) -> List[LangPair]:
        try:
            return list(self.splits[split].keys())
        except KeyError:
            raise ValueError(
                f"`split` must be a valid split name, but the {self.dataset_name} dataset has no split named '{split}'."
            )


class _LangPairExample(TypedDict):
    """Represents a line read from a language pair file."""

    split: CString
    lang_pair: CString
    line_number: int
    source_tokens: Tensor
    target_tokens: Tensor


class _NllbDataPipelineFactory:
    dataset_name: str
    data_dir: Path
    tokenizer: TextTokenizer
    splits: Dict[str, Dict[LangPair, int]]
    gang: Gang
    train: bool
    max_seq_len: int
    max_num_tokens: int
    shuffle_window_size: int
    eval_batch_size: int
    num_prefetch: int

    def __init__(
        self,
        dataset_name: str,
        data_dir: Path,
        tokenizer: TextTokenizer,
        splits: Dict[str, Dict[LangPair, int]],
        gang: Gang,
        train: bool,
        max_seq_len: int,
        max_num_tokens: int,
        shuffle_window_size: int,
        eval_batch_size: int,
        num_prefetch: int,
    ) -> None:
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.splits = splits
        self.gang = gang
        self.train = train
        self.max_seq_len = max_seq_len
        self.max_num_tokens = max_num_tokens
        self.shuffle_window_size = shuffle_window_size
        self.eval_batch_size = eval_batch_size
        self.num_prefetch = num_prefetch

    def __call__(self) -> DataPipeline:
        lang_pair_pipelines: List[DataPipeline] = []

        # The number of examples to be read per language pair.
        lang_pair_sizes: List[int] = []

        # The total number of examples to be read.
        total_size = 0

        for split_name, lang_pairs in self.splits.items():
            for lang_pair, size in lang_pairs.items():
                if size == 0:
                    raise DatasetError(
                        f"The '{split_name}' split of the {self.dataset_name} dataset has no examples for the language pair '{lang_pair}'."
                    )

                pipeline = self._create_lang_pair_pipeline(split_name, lang_pair)

                lang_pair_pipelines.append(pipeline)

                lang_pair_sizes.append(size)

                total_size += size

        if self.train:
            # Sample the language pairs based on their size.
            pipeline_builder = DataPipeline.sample(
                lang_pair_pipelines, weights=[s / total_size for s in lang_pair_sizes]
            )
        else:
            pipeline_builder = DataPipeline.concat(lang_pair_pipelines)

        return self._build_pipeline(pipeline_builder)

    def _create_lang_pair_pipeline(
        self, split_name: str, lang_pair: LangPair
    ) -> DataPipeline:
        source_lang, target_lang = lang_pair

        source_file, target_file = self._get_lang_pair_files(
            split_name, source_lang, target_lang
        )

        source_pipeline_builder = read_text(source_file, rtrim=True, memory_map=True)
        target_pipeline_builder = read_text(target_file, rtrim=True, memory_map=True)

        source_pipeline_builder.shard(self.gang.rank, self.gang.size)
        target_pipeline_builder.shard(self.gang.rank, self.gang.size)

        # Initialize the token encoders for the source and target languages.
        source_encoder_mode = "source"

        # Check if we have a train split with a specific data source.
        if split_name.startswith("train_"):
            source_encoder_mode = f"{source_encoder_mode}_{split_name[6:]}"

        source_encoder = self.tokenizer.create_encoder(
            task="translation", lang=source_lang, mode=source_encoder_mode
        )

        target_encoder = self.tokenizer.create_encoder(
            task="translation", lang=target_lang, mode="target"
        )

        source_pipeline_builder.map(
            source_encoder, num_parallel_calls=num_parallel_calls
        )
        target_pipeline_builder.map(
            target_encoder, num_parallel_calls=num_parallel_calls
        )

        source_pipeline = source_pipeline_builder.and_return()
        target_pipeline = target_pipeline_builder.and_return()

        # Include the language pair name and the line number with each example
        # for troubleshooting.
        split = DataPipeline.constant(split_name).and_return()

        lang_pair_ = DataPipeline.constant(str(lang_pair)).and_return()

        line_number = DataPipeline.count(
            start=self.gang.rank, step=self.gang.size
        ).and_return()

        # Zip the source and target pipelines along with the pseudo pipelines
        # into one.
        names = ["split", "lang_pair", "line_number", "source_tokens", "target_tokens"]

        pipeline_builder = DataPipeline.zip(
            [split, lang_pair_, line_number, source_pipeline, target_pipeline],
            names=names,
        )

        return pipeline_builder.and_return()

    def _get_lang_pair_files(
        self, split_name: str, source_lang: str, target_lang: str
    ) -> Tuple[Path, Path]:
        source_filename = f"{split_name}.{source_lang}-{target_lang}.{source_lang}"
        target_filename = f"{split_name}.{source_lang}-{target_lang}.{target_lang}"

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

    def _build_pipeline(self, pipeline_builder: DataPipelineBuilder) -> DataPipeline:
        if self.train:
            # Shuffle examples.
            if self.shuffle_window_size > 0:
                pipeline_builder.shuffle(self.shuffle_window_size)

            # Bucket by the length of the source or target sequence. The longer
            # one will be considered the length of the example.
            bucket_sizes = create_bucket_sizes(
                self.max_num_tokens, self.max_seq_len, min_seq_len=4
            )

            # Note that pipeline sampling and bucketing by length introduce non-
            # determinism in the number of examples read from the dataset. This
            # means, when used over a sharded data pipeline, they can cause the
            # training to hang in a distributed setting. One should ideally set
            # a safe upper limit on the number of examples to be read.
            pipeline_builder.bucket_by_length(
                bucket_sizes,
                selector="source_tokens,target_tokens",
                skip_long_examples=True,
            )
        else:
            pipeline_builder.bucket(self.eval_batch_size)

        # Collate bucketed examples into a batch.
        collater = Collater(pad_value=self.tokenizer.vocab_info.pad_idx)

        pipeline_builder.map(collater, num_parallel_calls=num_parallel_calls)

        # Prefetch examples in a background thread.
        if self.num_prefetch > 0:
            pipeline_builder.prefetch(self.num_prefetch)

        # And, return batches as `Seq2SeqBatch` instances.
        example_to_batch = partial(self._example_to_batch, device=self.gang.device)

        return pipeline_builder.map(example_to_batch).and_return()

    @staticmethod
    def _example_to_batch(example: Dict[str, Any], device: Device) -> Seq2SeqBatch:
        source_data = cast(SequenceData, example["source_tokens"])
        target_data = cast(SequenceData, example["target_tokens"])

        source_seqs, source_padding_mask = get_seqs_and_padding_mask(
            source_data, device
        )
        target_seqs, target_padding_mask = get_seqs_and_padding_mask(
            target_data, device
        )

        return Seq2SeqBatch(
            source_seqs, source_padding_mask, target_seqs, target_padding_mask, example
        )


@final
class NllbDatasetLoader(StandardDatasetLoader[NllbDataset]):
    """Loads multilingual NLLB text datasets."""

    @finaloverride
    def _load(self, path: Path, card: AssetCard) -> NllbDataset:
        """Extract the splits and their language pairs from the card."""
        splits = {}

        splits_field = card.field("splits")

        for split_name in splits_field.as_dict(dict).keys():
            lang_pairs = {}

            for key, size in splits_field.field(split_name).as_dict(int).items():
                try:
                    source_lang, target_lang = key.split("-")
                except ValueError as ex:
                    raise AssetCardError(
                        f"The items of the field 'splits.{split_name}' of the asset card '{card.name}' must represent language pairs, but '{key}' does not represent a language pair."
                    ) from ex

                lang_pair = LangPair(source_lang, target_lang)

                lang_pairs[lang_pair] = size

            splits[split_name] = lang_pairs

        return NllbDataset(card.name, path, splits)


load_nllb_dataset = NllbDatasetLoader(default_asset_store, default_download_manager)

load_multilingual_text_dataset.register_loader("nllb", load_nllb_dataset)
