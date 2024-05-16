# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, NamedTuple, Optional, Sequence

from fairseq2.data.text import TextTokenizer
from fairseq2.datasets.data_reader import DataReader
from fairseq2.datasets.loader import DelegatingDatasetLoader
from fairseq2.gang import Gang
from fairseq2.models.seq2seq import Seq2SeqBatch


class LangPair(NamedTuple):
    """Represents the language pair of a parallel corpus."""

    source_lang: str
    """The source language code."""

    target_lang: str
    """The target language code."""

    def __repr__(self) -> str:
        return f"{self.source_lang}-{self.target_lang}"


class ParallelTextDataset(ABC):
    """Represents a parallel text dataset."""

    @abstractmethod
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        max_seq_len: int,
        max_num_tokens: int,
        *,
        lang_pairs: Optional[Sequence[LangPair]] = None,
        sample: bool = False,
        shuffle_window_size: int = 1,
        num_repeats: Optional[int] = 1,
        num_accumulate: int = 1,
        num_prefetch: int = 0,
        seed: int = 2,
        **extras: Any,
    ) -> DataReader[Seq2SeqBatch]:
        """Create a dataset reader.

        :param split:
            The split to read.
        :param tokenizer:
            The tokenizer to encode text.
        :param gang:
            The gang over which to shard the dataset.
        :param max_seq_len:
            The maximum sequence length of each example. Examples longer than
            this value will be dropped.
        :param max_num_tokens:
            The maximum number of tokens in each batch.
        :param lang_pairs:
            The language pairs to read. If ``None``, all pairs will be read.
        :param sample:
            If ``True``, language pair corpora will be sampled in proportion to
            their size.
        :param shuffle_window_size:
            The size of the shuffle window. If ``1``, no shuffling is performed;
            if ``0``, performs true shuffling by loading the entire dataset.
        :param num_repeats:
            The dataset will be repeatedly read this many times. If ``None``, it
            will be read indefinitely.
        :param num_accumulate:
            The number of batches to accumulate in each iteration. Typically
            used with gradient accumulation during training.
        :param num_prefetch:
            The number of batches to prefetch in background.
        :param seed:
            The seed to initialize the random number generators used internally.
        :param extras:
            The extra parameters specific to the dataset implementation.
        """

    @abstractmethod
    def splits(self) -> List[str]:
        """Return the list of splits."""

    @abstractmethod
    def lang_pairs(self, split: str) -> List[LangPair]:
        """Return the list of language pairs of ``split``."""


load_parallel_text_dataset = DelegatingDatasetLoader[ParallelTextDataset]()
