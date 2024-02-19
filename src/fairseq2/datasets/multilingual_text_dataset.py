# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, NamedTuple, Optional, Sequence

from fairseq2.assets import default_asset_store
from fairseq2.data import DataPipeline
from fairseq2.data.text import TextTokenizer
from fairseq2.datasets.loader import CompositeDatasetLoader
from fairseq2.gang import Gang


class LangPair(NamedTuple):
    """Represents a language pair to use in translation tasks."""

    source_lang: str
    """The source language code."""

    target_lang: str
    """The target language code."""

    def __repr__(self) -> str:
        return f"{self.source_lang}-{self.target_lang}"


class MultilingualTextDataset(ABC):
    """Represents a multilingual text dataset to use in translation tasks."""

    @abstractmethod
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
        :param shuffle_window_size:
            The size of the streaming shuffle window.
        :param eval_batch_size:
            The batch size for non-train splits.
        :param num_prefetch:
            The number of batches to prefetch in background.
        :param lang_pairs:
            The language pairs to read. If ``None``, all pairs will be read.
        """

    @abstractmethod
    def get_splits(self) -> List[str]:
        """Return the available splits of the dataset."""

    @abstractmethod
    def get_lang_pairs(self, split: str) -> List[LangPair]:
        """Returns the list of language pairs for ``split``."""


load_multilingual_text_dataset = CompositeDatasetLoader[MultilingualTextDataset](
    default_asset_store
)
