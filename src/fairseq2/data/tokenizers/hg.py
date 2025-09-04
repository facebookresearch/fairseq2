# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, final

import torch
from torch import Tensor
from typing_extensions import override

try:
    from transformers import (  # type: ignore[import-not-found, attr-defined]
        AutoTokenizer,
        PreTrainedTokenizer,
    )
except ImportError:
    _has_hg_transformers = False
else:
    _has_hg_transformers = True

from fairseq2.data.tokenizers.family import TokenizerModelError
from fairseq2.data.tokenizers.tokenizer import TokenDecoder, TokenEncoder
from fairseq2.data.tokenizers.vocab_info import VocabularyInfo
from fairseq2.device import Device
from fairseq2.error import OperationalError
from fairseq2.utils.tensor import to_tensor

if TYPE_CHECKING:

    def load_hg_token_model(
        path: Path,
        *,
        unk_token: str | None = ...,
        bos_token: str | None = ...,
        eos_token: str | None = ...,
        pad_token: str | None = ...,
        boh_token: str | None = ...,
        eoh_token: str | None = ...,
    ) -> HuggingFaceTokenModel: ...

    @final
    class HuggingFaceTokenModel:
        _tok: PreTrainedTokenizer

        def encode(self, text: str) -> list[int]: ...

        def overwrite_chat_template(self, chat_template: str) -> None: ...

        def decode(
            self, token_indices: Sequence[int], skip_special_tokens: bool = False
        ) -> str: ...

        def decode_tensor(
            self, token_indices: Tensor, skip_special_tokens: bool = False
        ) -> str: ...

        def convert_tokens_to_ids(self, tokens: Sequence[str]) -> list[int]: ...

        def convert_ids_to_tokens(self, ids: Sequence[int]) -> list[str]: ...

        @property
        def vocab_info(self) -> VocabularyInfo: ...

else:

    def load_hg_token_model(
        path: Path,
        *,
        unk_token: str | None = None,
        bos_token: str | None = None,
        eos_token: str | None = None,
        pad_token: str | None = None,
        boh_token: str | None = None,
        eoh_token: str | None = None,
    ) -> HuggingFaceTokenModel:
        if not _has_hg_transformers:
            raise OperationalError(
                "Hugging Face Transformers is not found. Use `pip install transformers`."
            )

        try:
            tok = AutoTokenizer.from_pretrained(path)
        except RuntimeError as ex:
            msg = f"{path} is not a valid Hugging Face tokenizer model."

            raise TokenizerModelError(path, msg) from ex

        vocab_size = len(tok)

        def maybe_index(token: str | None) -> int | None:
            if token:
                return tok.convert_tokens_to_ids(token)

            return None

        vocab_info = VocabularyInfo(
            vocab_size,
            unk_idx=maybe_index(unk_token),
            bos_idx=maybe_index(bos_token),
            eos_idx=maybe_index(eos_token),
            pad_idx=maybe_index(pad_token),
            boh_idx=maybe_index(boh_token),
            eoh_idx=maybe_index(eoh_token),
        )

        return HuggingFaceTokenModel(tok, vocab_info)

    @final
    class HuggingFaceTokenModel:
        def __init__(
            self, tok: PreTrainedTokenizer, vocab_info: VocabularyInfo
        ) -> None:
            self._tok = tok
            self._vocab_info = vocab_info

        def encode(self, text: str) -> list[int]:
            return self._tok.encode(text, add_special_tokens=False)

        def overwrite_chat_template(self, chat_template: str) -> None:
            self._tok.chat_template = chat_template

        def decode(
            self, token_indices: Sequence[int], skip_special_tokens: bool = False
        ) -> str:
            return self._tok.decode(
                token_indices, skip_special_tokens=skip_special_tokens
            )

        def decode_tensor(
            self, token_indices: Tensor, skip_special_tokens: bool = False
        ) -> str:
            return self._tok.decode(
                token_indices, skip_special_tokens=skip_special_tokens
            )

        def convert_tokens_to_ids(self, tokens: Sequence[str]) -> list[int]:
            return self._tok.convert_tokens_to_ids(tokens)

        def convert_ids_to_tokens(self, ids: Sequence[int]) -> list[str]:
            return self._tok.convert_ids_to_tokens(ids)

        @property
        def vocab_info(self) -> VocabularyInfo:
            return self._vocab_info


@final
class HuggingFaceTokenEncoder(TokenEncoder):
    def __init__(
        self,
        model: HuggingFaceTokenModel,
        *,
        prefix_tokens: list[str] | None = None,
        suffix_tokens: list[str] | None = None,
        device: Device | None = None,
        pin_memory: bool = False,
    ) -> None:
        """
        :param prefix_tokens: The prefix tokens to encode with input text.
        :param suffix_tokens: The suffix tokens to encode with input text.
        :param device: The device on which to construct tensors.
        :param pin_memory: If ``True``, uses pinned memory for tensors.
        """
        self._model = model

        self._prefix_index_pt: Tensor | None
        self._suffix_index_pt: Tensor | None

        # Prefix
        if prefix_tokens:
            self._prefix_indices = model.convert_tokens_to_ids(prefix_tokens)

            self._prefix_index_pt = to_tensor(
                self._prefix_indices, dtype=torch.int64, device=device
            )
        else:
            self._prefix_indices = []

            self._prefix_index_pt = None

        # Suffix
        if suffix_tokens:
            self._suffix_indices = model.convert_tokens_to_ids(suffix_tokens)

            self._suffix_index_pt = to_tensor(
                self._suffix_indices, dtype=torch.int64, device=device
            )
        else:
            self._suffix_indices = []

            self._suffix_index_pt = None

        self._device = device

        self._pin_memory = pin_memory

    @override
    def __call__(self, text: str) -> Tensor:
        indices = self._model.encode(text)

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

        return self._model.convert_ids_to_tokens(indices)

    def apply_chat_template(self, chat: Any, **kwargs: Any) -> Any:
        return self._model._tok.apply_chat_template(chat, **kwargs)

    @property
    @override
    def prefix_indices(self) -> Tensor | None:
        return self._prefix_index_pt

    @property
    @override
    def suffix_indices(self) -> Tensor | None:
        return self._suffix_index_pt


@final
class HuggingFaceTokenDecoder(TokenDecoder):
    def __init__(
        self, model: HuggingFaceTokenModel, *, skip_special_tokens: bool = False
    ) -> None:
        self._model = model
        self._skip_special_tokens = skip_special_tokens

    @override
    def __call__(self, token_indices: Tensor) -> str:
        if token_indices.ndim != 1:
            raise ValueError(
                f"`token_indices` must be one dimensional, but has {token_indices.ndim} dimensions instead."
            )

        return self._model.decode_tensor(
            token_indices, skip_special_tokens=self._skip_special_tokens
        )

    @override
    def decode_from_tokens(self, tokens: Sequence[str]) -> str:
        indices = self._model.convert_tokens_to_ids(tokens)

        return self._model.decode(
            indices, skip_special_tokens=self._skip_special_tokens
        )
