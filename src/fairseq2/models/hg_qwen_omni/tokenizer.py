# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HuggingFace tokenizer integration for fairseq2.

This module provides a bridge between HuggingFace tokenizers and fairseq2's
tokenizer interface. It allows you to use any HuggingFace tokenizer within
the fairseq2 ecosystem while maintaining compatibility with fairseq2's
training and inference pipelines.

Classes:
    HgTokenizerConfig: Configuration for HuggingFace tokenizers
    HgTokenizer: Wrapper class that adapts HuggingFace tokenizers

Functions:
    load_hg_tokenizer: Load a HuggingFace tokenizer from path and config

Example:
    Load a GPT-2 tokenizer::

        from fairseq2.models.hg_qwen_omni.tokenizer import (
            load_hg_tokenizer,
            HgTokenizerConfig,
        )

        config = HgTokenizerConfig()
        tokenizer = load_hg_tokenizer("gpt2", config)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing_extensions import override

from fairseq2.data.tokenizers import (
    TokenDecoder,
    TokenEncoder,
    Tokenizer,
    VocabularyInfo,
)
from fairseq2.data.tokenizers.hg import (
    HuggingFaceTokenDecoder,
    HuggingFaceTokenEncoder,
    HuggingFaceTokenModel,
    load_hg_token_model,
)
from fairseq2.device import Device


@dataclass(kw_only=True)
class HgTokenizerConfig:
    """Configuration for HuggingFace tokenizers."""

    unk_token: str | None = None
    """The unknown token."""

    bos_token: str | None = None
    """The beginning-of-sequence token."""

    eos_token: str | None = None
    """The end-of-sequence token."""

    pad_token: str | None = None
    """The padding token."""

    boh_token: str | None = None
    """The beginning-of-head token."""

    eoh_token: str | None = None
    """The end-of-head token."""


class HgTokenizer(Tokenizer):
    """HuggingFace tokenizer adapter for fairseq2.

    This class wraps a HuggingFace tokenizer to make it compatible with
    fairseq2's Tokenizer interface. It provides access to both fairseq2
    tokenizer methods and the underlying HuggingFace tokenizer.

    Example:
        Create a tokenizer from a model::

            model = load_hg_token_model("gpt2")
            tokenizer = HgTokenizer(model)

            # Use fairseq2 interface
            tokens = tokenizer.encode("Hello world")
            text = tokenizer.decode(tokens)

            # Access underlying HuggingFace tokenizer
            hf_tokenizer = tokenizer.raw
    """

    def __init__(self, model: HuggingFaceTokenModel) -> None:
        self._model = model
        self._encoder: TokenEncoder | None = None
        self._decoder: TokenDecoder | None = None

    @override
    def create_encoder(
        self,
        *,
        task: str | None = None,
        lang: str | None = None,
        mode: str | None = None,
        device: Device | None = None,
        pin_memory: bool = False,
    ) -> TokenEncoder:
        return self.create_raw_encoder(device=device, pin_memory=pin_memory)

    @override
    def create_raw_encoder(
        self, *, device: Device | None = None, pin_memory: bool = False
    ) -> TokenEncoder:
        if self._encoder is not None:
            return self._encoder
        self._encoder = HuggingFaceTokenEncoder(
            self._model, device=device, pin_memory=pin_memory
        )
        return self._encoder

    @override
    def create_decoder(self, *, skip_special_tokens: bool = False) -> TokenDecoder:
        if self._decoder is not None:
            return self._decoder
        self._decoder = HuggingFaceTokenDecoder(
            self._model, skip_special_tokens=skip_special_tokens
        )
        return self._decoder

    def encode(
        self, text: str, *, device: Device | None = None, pin_memory: bool = False
    ) -> Tensor:
        encoder = self.create_raw_encoder(device=device, pin_memory=pin_memory)
        return encoder(text)

    def decode(
        self, token_indices: Tensor, *, skip_special_tokens: bool = False
    ) -> str:
        decoder = self.create_decoder(skip_special_tokens=skip_special_tokens)
        return decoder(token_indices)

    # TODO: maybe a better way is just fall back to properties of
    # self._model._tok if it's not overridden in this fs2 Tokenizer class..?
    def convert_tokens_to_ids(self, tokens: list[str] | str) -> int | list[int]:
        return self._model._tok.convert_tokens_to_ids(tokens)

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        return self._model.vocab_info

    @property
    def unk_token(self) -> str | None:
        if hasattr(self._model._tok, "unk_token"):
            return str(self._model._tok.unk_token)
        return None

    @property
    def bos_token_id(self) -> str | None:
        if hasattr(self._model._tok, "bos_token_id"):
            return str(self._model._tok.bos_token_id)
        return None

    @property
    def bos_token(self) -> str | None:
        if hasattr(self._model._tok, "bos_token"):
            return str(self._model._tok.bos_token)
        return None

    @property
    def eos_token_id(self) -> str | None:
        if hasattr(self._model._tok, "eos_token_id"):
            return str(self._model._tok.eos_token_id)
        return None

    @property
    def eos_token(self) -> str | None:
        if hasattr(self._model._tok, "eos_token"):
            return str(self._model._tok.eos_token)
        return None

    @property
    def pad_token_id(self) -> str | None:
        if hasattr(self._model._tok, "pad_token_id"):
            return str(self._model._tok.pad_token_id)
        return None

    @property
    def pad_token(self) -> str | None:
        if hasattr(self._model._tok, "pad_token"):
            return str(self._model._tok.pad_token)
        return None

    @property
    def boh_token(self) -> str | None:
        if hasattr(self._model._tok, "boh_token"):
            return str(self._model._tok.boh_token)
        return None

    @property
    def eoh_token(self) -> str | None:
        if hasattr(self._model._tok, "eoh_token"):
            return str(self._model._tok.eoh_token)
        return None

    @property
    def chat_template(self) -> str | None:
        if hasattr(self._model._tok, "chat_template"):
            return str(self._model._tok.chat_template)
        return None

    @property
    def raw(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        return self._model._tok

    @property
    def model(self) -> HuggingFaceTokenModel:
        return self._model


def load_hg_tokenizer(path: Path, config: HgTokenizerConfig) -> HgTokenizer:
    """
    Load a HuggingFace tokenizer.

    :param config: Tokenizer configuration
    :returns: HgTokenizer instance
    """

    # Load the HuggingFace token model
    model = load_hg_token_model(
        path,
        unk_token=config.unk_token,
        bos_token=config.bos_token,
        eos_token=config.eos_token,
        pad_token=config.pad_token,
        boh_token=config.boh_token,
        eoh_token=config.eoh_token,
    )

    return HgTokenizer(model)
