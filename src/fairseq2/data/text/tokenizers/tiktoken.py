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
from tiktoken import Encoding
from tiktoken.load import load_tiktoken_bpe
from torch import Tensor
from typing_extensions import override

from fairseq2.data import VocabularyInfo
from fairseq2.data.text.tokenizers import (
    AbstractTextTokenizer,
    TextTokenDecoder,
    TextTokenEncoder,
)
from fairseq2.typing import Device


class TiktokenTokenizer(AbstractTextTokenizer):
    """Represents a tiktoken tokenizer."""

    _encoding: Encoding
    _num_bpe_tokens: int

    def __init__(
        self,
        path: Path,
        split_regex: str,
        *,
        unk_token: str | None = None,
        bos_token: str | None = None,
        eos_token: str | None = None,
        pad_token: str | None = None,
        boh_token: str | None = None,
        eoh_token: str | None = None,
        special_tokens: Sequence[str] | None = None,
    ) -> None:
        """
        :param path: The path to the tiktoken BPE file.
        :param split_regex: The regex pattern string that is used to split the
            input text.
        :param unk_token: The token that represents an unknown element.
        :param bos_token: The token that represents the beginning of a sequence.
        :param eos_token: The token that represents the end of a sequence.
        :param pad_token: The token that is used to pad a sequence.
        :param boh_token: The token that represents the beginning of a header.
        :param eoh_token: The token that represents the end of a header.
        :param special_tokens: The extra special tokens to include in the
            tokenizer.
        """
        tokens = load_tiktoken_bpe(str(path))

        num_tokens = len(tokens)

        self._num_bpe_tokens = num_tokens

        if special_tokens:
            special_token_map = {
                token: num_tokens + i for i, token in enumerate(special_tokens)
            }
        else:
            special_token_map = {}

        self._encoding = Encoding(
            name=path.stem,
            pat_str=split_regex,
            mergeable_ranks=tokens,
            special_tokens=special_token_map,
        )

        def maybe_index(token: str | None) -> int | None:
            if token:
                return self._encoding.encode_single_token(token)

            return None

        vocab_info = VocabularyInfo(
            self._encoding.n_vocab,
            unk_idx=maybe_index(unk_token),
            bos_idx=maybe_index(bos_token),
            eos_idx=maybe_index(eos_token),
            pad_idx=maybe_index(pad_token),
            boh_idx=maybe_index(boh_token),
            eoh_idx=maybe_index(eoh_token),
        )

        super().__init__(vocab_info)

    @override
    def create_raw_encoder(
        self, *, device: Device | None = None, pin_memory: bool = False
    ) -> TiktokenEncoder:
        return TiktokenEncoder(self._encoding, device=device, pin_memory=pin_memory)

    @override
    def create_decoder(self) -> TiktokenDecoder:
        return TiktokenDecoder(self._encoding, self._num_bpe_tokens)


@final
class TiktokenEncoder(TextTokenEncoder):
    """Represents a tiktoken decoder."""

    _prefix_indices: list[int]
    _suffix_indices: list[int]
    _prefix_index_tensor: Tensor | None
    _suffix_index_tensor: Tensor | None
    _device: Device | None
    _pin_memory: bool

    def __init__(
        self,
        encoding: Encoding,
        *,
        prefix_tokens: Sequence[str] | None = None,
        suffix_tokens: Sequence[str] | None = None,
        device: Device | None = None,
        pin_memory: bool = False,
    ) -> None:
        """
        :param encoding:
            The tiktoken :class:`Encoding` object.
        :param prefix_tokens:
            The prefix tokens to encode with input text.
        :param suffix_tokens:
            The suffix tokens to encode with input text.
        :param device:
            The device on which to construct tensors.
        :param pin_memory:
            If ``True``, uses pinned memory while constructing tensors.
        """
        self._encoding = encoding

        # Prefix
        if prefix_tokens:
            self._prefix_indices = [
                encoding.encode_single_token(t) for t in prefix_tokens
            ]

            self._prefix_index_tensor = torch.tensor(
                self._prefix_indices, dtype=torch.int64, device=device
            )
        else:
            self._prefix_indices = []

            self._prefix_index_tensor = None

        # Suffix
        if suffix_tokens:
            self._suffix_indices = [
                encoding.encode_single_token(t) for t in suffix_tokens
            ]

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
        indices = self._encoding.encode(text, allowed_special="all")

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

        b = self._encoding.decode_tokens_bytes(indices)

        return [e.decode("utf-8") for e in b]

    @property
    @override
    def prefix_indices(self) -> Tensor | None:
        return self._prefix_index_tensor

    @property
    @override
    def suffix_indices(self) -> Tensor | None:
        return self._suffix_index_tensor


@final
class TiktokenDecoder(TextTokenDecoder):
    """Represents a tiktoken decoder."""

    _encoding: Encoding
    _num_bpe_tokens: int

    def __init__(self, encoding: Encoding, num_bpe_tokens: int) -> None:
        """
        :param encoding:
            The tiktoken :class:`Encoding` object.
        param num_bpe_tokens:
            The number of byte-pair encoding tokens, excluding special tokens.
        """
        self._encoding = encoding
        self._num_bpe_tokens = num_bpe_tokens

    @override
    def __call__(self, token_indices: Tensor) -> str:
        if token_indices.dim() != 1:
            raise ValueError(
                f"`token_indices` must be one dimensional, but has {token_indices.dim()} dimensions instead."
            )

        # Do not decode special tokens.
        indices = [i for i in token_indices.tolist() if i < self._num_bpe_tokens]

        return self._encoding.decode(indices)

    @override
    def decode_from_tokens(self, tokens: Sequence[str]) -> str:
        indices = [self._encoding.encode_single_token(t) for t in tokens]

        # Do not decode special tokens.
        indices = [i for i in indices if i < self._num_bpe_tokens]

        return self._encoding.decode(indices)
