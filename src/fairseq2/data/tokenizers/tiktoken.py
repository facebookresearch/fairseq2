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

from fairseq2.data.tokenizers.family import TokenizerModelError
from fairseq2.data.tokenizers.tokenizer import TokenDecoder, TokenEncoder
from fairseq2.data.tokenizers.vocab_info import VocabularyInfo
from fairseq2.device import Device
from fairseq2.utils.tensor import to_tensor


def load_tiktoken_model(
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
) -> TiktokenModel:
    """
    :param path: The path to the tiktoken BPE file.
    :param split_regex: The regex pattern used to split the input text.
    :param unk_token: The token that represents an unknown element.
    :param bos_token: The token that represents the beginning of a sequence.
    :param eos_token: The token that represents the end of a sequence.
    :param pad_token: The token that is used to pad a sequence.
    :param boh_token: The token that represents the beginning of a header.
    :param eoh_token: The token that represents the end of a header.
    :param special_tokens: The special tokens to include in the tokenizer.
    """
    try:
        tokens = load_tiktoken_bpe(str(path))
    except RuntimeError as ex:
        msg = f"{path} is not a valid tiktoken model."

        raise TokenizerModelError(path, msg) from ex

    num_tokens = len(tokens)

    if special_tokens:
        special_token_map = {
            token: num_tokens + i for i, token in enumerate(special_tokens)
        }
    else:
        special_token_map = {}

    encoding = Encoding(
        name=path.stem,
        pat_str=split_regex,
        mergeable_ranks=tokens,
        special_tokens=special_token_map,
    )

    def maybe_index(token: str | None) -> int | None:
        if token:
            return encoding.encode_single_token(token)

        return None

    vocab_info = VocabularyInfo(
        encoding.n_vocab,
        unk_idx=maybe_index(unk_token),
        bos_idx=maybe_index(bos_token),
        eos_idx=maybe_index(eos_token),
        pad_idx=maybe_index(pad_token),
        boh_idx=maybe_index(boh_token),
        eoh_idx=maybe_index(eoh_token),
    )

    return TiktokenModel(encoding, num_tokens, vocab_info)


@final
class TiktokenModel:
    def __init__(
        self, encoding: Encoding, num_tokens: int, vocab_info: VocabularyInfo
    ) -> None:
        self._encoding = encoding
        self._num_bpe_tokens = num_tokens
        self._vocab_info = vocab_info

    @property
    def encoding(self) -> Encoding:
        return self._encoding

    @property
    def num_bpe_tokens(self) -> int:
        """The number of byte-pair encoding tokens, excluding special tokens."""
        return self._num_bpe_tokens

    @property
    def vocab_info(self) -> VocabularyInfo:
        return self._vocab_info


@final
class TiktokenEncoder(TokenEncoder):
    def __init__(
        self,
        model: TiktokenModel,
        *,
        prefix_tokens: Sequence[str] | None = None,
        suffix_tokens: Sequence[str] | None = None,
        device: Device | None = None,
        pin_memory: bool = False,
    ) -> None:
        """
        :param encoding: The tiktoken :class:`Encoding` object.
        :param prefix_tokens: The prefix tokens to encode with input text.
        :param suffix_tokens: The suffix tokens to encode with input text.
        :param device: The device on which to construct tensors.
        :param pin_memory: If ``True``, uses pinned memory for tensors.
        """
        self._encoding = model.encoding

        self._prefix_indices_pt: Tensor | None
        self._suffix_indices_pt: Tensor | None

        # Prefix
        if prefix_tokens:
            self._prefix_indices = [
                self._encoding.encode_single_token(t) for t in prefix_tokens
            ]

            self._prefix_indices_pt = to_tensor(
                self._prefix_indices, dtype=torch.int64, device=device
            )
        else:
            self._prefix_indices = []

            self._prefix_indices_pt = None

        # Suffix
        if suffix_tokens:
            self._suffix_indices = [
                self._encoding.encode_single_token(t) for t in suffix_tokens
            ]

            self._suffix_indices_pt = to_tensor(
                self._suffix_indices, dtype=torch.int64, device=device
            )
        else:
            self._suffix_indices = []

            self._suffix_indices_pt = None

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
        return self._prefix_indices_pt

    @property
    @override
    def suffix_indices(self) -> Tensor | None:
        return self._suffix_indices_pt


@final
class TiktokenDecoder(TokenDecoder):
    def __init__(
        self, model: TiktokenModel, *, skip_special_tokens: bool = False
    ) -> None:
        self._encoding = model.encoding
        self._num_bpe_tokens = model.num_bpe_tokens
        self._skip_special_tokens = skip_special_tokens

    @override
    def __call__(self, token_indices: Tensor) -> str:
        if token_indices.ndim != 1:
            raise ValueError(
                f"`token_indices` must be one dimensional, but has {token_indices.ndim} dimensions instead."
            )

        indices = token_indices.tolist()

        if self._skip_special_tokens:
            indices = [i for i in indices if i < self._num_bpe_tokens]

        return self._encoding.decode(indices)

    @override
    def decode_from_tokens(self, tokens: Sequence[str]) -> str:
        indices = [self._encoding.encode_single_token(t) for t in tokens]

        if self._skip_special_tokens:
            indices = [i for i in indices if i < self._num_bpe_tokens]

        return self._encoding.decode(indices)
