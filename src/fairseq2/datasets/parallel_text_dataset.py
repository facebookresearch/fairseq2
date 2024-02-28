# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    cast,
    final,
)

from fairseq2.assets import default_asset_store
from fairseq2.data import DataPipeline, SequenceData
from fairseq2.data.text import TextTokenizer
from fairseq2.datasets.loader import DelegatingDatasetLoader
from fairseq2.datasets.utils import all_eod
from fairseq2.gang import Gang
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.typing import Device, override

logger = logging.getLogger(__name__)


class LangPair(NamedTuple):
    """Represents a language pair in parallel text."""

    source_lang: str
    """The source language code."""

    target_lang: str
    """The target language code."""

    def __repr__(self) -> str:
        return f"{self.source_lang}-{self.target_lang}"


class ParallelTextDataset(ABC):
    """Represents a parallel text dataset."""

    @abstractmethod
    def read(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        max_seq_len: int,
        max_num_tokens: int,
        *,
        bucket_by_length: bool = False,
        sample: bool = False,
        shuffle_window_size: int = 0,
        num_prefetch: int = 0,
        num_accumulate: int = 1,
        lang_pairs: Optional[Sequence[LangPair]] = None,
    ) -> Iterator[List[Seq2SeqBatch]]:
        """Read the dataset.

        :param split:
            The split to read.
        :param tokenizer:
            The tokenizer to encode text.
        :param gang:
            The gang to shard the dataset.
        :param max_seq_len:
            The maximum sequence length of each example. Examples longer than
            this value will be silently dropped.
        :param max_num_tokens:
            The maximum number of tokens in each batch.
        :param bucket_by_length:
            If ``True``, examples will be bucketed by their length.
        :param sample:
            If ``True``, language pair corpora will be sampled in proportion to
            their size.
        :param shuffle_window_size:
            The size of the streaming shuffle window.
        :param num_prefetch:
            The number of batches to prefetch in background.
        :param num_accumulate:
            The number of batches to accumulate in each iteration. Typically
            used with gradient accumulation during training.
        :param lang_pairs:
            The language pairs to read. If ``None``, all pairs will be read.
        """

    @abstractmethod
    def splits(self) -> List[str]:
        """Return the list of splits."""

    @abstractmethod
    def lang_pairs(self, split: str) -> List[LangPair]:
        """Return the list of language pairs of ``split``."""

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """The name of the dataset."""


class AbstractParallelTextDataset(ParallelTextDataset):
    """Provides a skeletal implementation of :class:`ParallelTextDataset`."""

    _dataset_name: str

    def __init__(self, dataset_name: str) -> None:
        self._dataset_name = dataset_name

    @final
    @override
    def read(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        max_seq_len: int,
        max_num_tokens: int,
        *,
        bucket_by_length: bool = False,
        sample: bool = False,
        shuffle_window_size: int = 0,
        num_prefetch: int = 0,
        num_accumulate: int = 1,
        lang_pairs: Optional[Sequence[LangPair]] = None,
    ) -> Iterator[List[Seq2SeqBatch]]:
        pipeline = self._build_pipeline(
            split,
            tokenizer,
            gang,
            max_seq_len,
            max_num_tokens,
            bucket_by_length,
            sample,
            shuffle_window_size,
            num_prefetch,
            lang_pairs,
        )

        eod = False

        pipeline_iter = iter(pipeline)

        while not eod:
            batches = []

            for _ in range(num_accumulate):
                try:
                    example = next(pipeline_iter)
                except StopIteration:
                    break

                batch = self._example_to_batch(example, gang.device)

                batches.append(batch)

            eod = len(batches) != num_accumulate

            # When the pipeline is sharded, sampling and bucketing by length
            # can lead to unpredictability in the number of examples read in
            # each process. So, it is important to ensure that all processes
            # are in sync about the end of the data. If this is not the case,
            # a training loop may become stuck.
            if sample or bucket_by_length:
                eod = all_eod(eod, gang, logger)

            if not eod:
                yield batches

    @abstractmethod
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
        ...

    @staticmethod
    def _example_to_batch(example: Dict[str, Any], device: Device) -> Seq2SeqBatch:
        source_data = cast(SequenceData, example["source_indices"])
        target_data = cast(SequenceData, example["target_indices"])

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
    @property
    @override
    def dataset_name(self) -> str:
        return self._dataset_name


load_parallel_text_dataset = DelegatingDatasetLoader[ParallelTextDataset](
    default_asset_store
)
