# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import final

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.data import VocabularyInfo
from fairseq2.data.text.tokenizers import (
    TextTokenDecoder,
    TextTokenEncoder,
)
from fairseq2.typing import Device

try:
    from transformers import AutoTokenizer
except ImportError:
    raise RuntimeError(
        "transformers library is required to use HF tokenizers. Install it via `pip install transformers`."
    )


@final
class HuggingfaceTokenizerEncoder(TextTokenEncoder):
    """Represents a tiktoken decoder."""

    _tokenizer: AutoTokenizer
    _prefix_indices: list[int]
    _suffix_indices: list[int]
    _prefix_index_tensor: Tensor | None
    _suffix_index_tensor: Tensor | None
    _device: Device | None
    _pin_memory: bool

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        *,
        prefix_tokens: Sequence[str] | None = None,
        suffix_tokens: Sequence[str] | None = None,
        device: Device | None = None,
        pin_memory: bool = False,
    ) -> None:
        """
        :param tokenizer:
            The huggingface :class:`AutoTokenizer` object.
        :param prefix_tokens:
            The prefix tokens to encode with input text.
        :param suffix_tokens:
            The suffix tokens to encode with input text.
        :param device:
            The device on which to construct tensors.
        :param pin_memory:
            If ``True``, uses pinned memory while constructing tensors.
        """
        self._tokenizer = tokenizer

        # Prefix
        if prefix_tokens:
            self._prefix_indices = self._tokenizer.convert_tokens_to_ids(prefix_tokens)

            self._prefix_index_tensor = torch.tensor(
                self._prefix_indices, dtype=torch.int64, device=device
            )
        else:
            self._prefix_indices = []

            self._prefix_index_tensor = None

        # Suffix
        if suffix_tokens:
            self._suffix_indices = self._tokenizer.convert_tokens_to_ids(suffix_tokens)

            self._suffix_index_tensor = torch.tensor(
                self._suffix_indices, dtype=torch.int64, device=device
            )
        else:
            self._suffix_indices = []

            self._suffix_index_tensor = None

        self._device = device
        self._pin_memory = pin_memory

    @override
    def __call__(self, text: str) -> Tensor:
        # fairseq2 tokenizer adds special tokens on its own
        indices = self._tokenizer.encode(text, add_special_tokens=False)

        if self._prefix_indices:
            indices = self._prefix_indices + indices

        if self._suffix_indices:
            indices.extend(self._suffix_indices)

        return torch.tensor(
            indices, dtype=torch.int64, device=self._device, pin_memory=self._pin_memory
        )

    @override
    def encode_as_tokens(self, text: str) -> list[str]:
        indices = self(text).tolist()

        tokens = self._tokenizer.convert_tds_to_tokens(indices)

        return tokens

    @property
    @override
    def prefix_indices(self) -> Tensor | None:
        return self._prefix_index_tensor

    @property
    @override
    def suffix_indices(self) -> Tensor | None:
        return self._suffix_index_tensor


@final
class HuggingfaceTokenizerDecoder(TextTokenDecoder):
    """Represents a tiktoken decoder."""

    _tokenizer: AutoTokenizer

    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self._tokenizer = tokenizer

    @override
    def __call__(self, token_indices: Tensor) -> str:
        if token_indices.dim() != 1:
            raise ValueError(
                f"`token_indices` must be one dimensional, but has {token_indices.dim()} dimensions instead."
            )

        return self._tokenizer.decode(token_indices)

    @override
    def decode_from_tokens(self, tokens: Sequence[str]) -> str:
        indices = self._tokenizer.convert_tokens_to_ids(tokens)

        return self._tokenizer.decode(indices)
