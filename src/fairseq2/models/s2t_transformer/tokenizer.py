# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, final

from typing_extensions import override

from fairseq2.data.tokenizers import (
    TokenDecoder,
    TokenEncoder,
    Tokenizer,
    VocabularyInfo,
)
from fairseq2.data.tokenizers.sentencepiece import (
    SentencePieceDecoder,
    SentencePieceEncoder,
    SentencePieceModel,
    get_sentencepiece_vocabulary_info,
    load_sentencepiece_model,
)
from fairseq2.device import Device
from fairseq2.error import NotSupportedError


@final
class S2TTransformerTokenizer(Tokenizer):
    """Represents an S2T Transformer tokenizer."""

    def __init__(
        self,
        model: SentencePieceModel,
        task: str,
        target_langs: Set[str],
        default_target_lang: str,
    ) -> None:
        """
        :param path:
            The path to the SentencePiece model file.
        :param task:
            The task for which to generate token indices. The valid values are
            'transcription' and 'translation'.
        :param target_langs:
            The list of supported target languages.
        :param default_target_lang:
            The fall-back language if no target language is specified.
        """
        if task != "transcription" and task != "translation":
            raise ValueError(
                f"`task` must be 'transcripton' or 'translation', but is '{task}' instead."
            )

        self._model = model
        self._task = task
        self._target_langs = target_langs
        self._default_target_lang = default_target_lang

        self._vocab_info = get_sentencepiece_vocabulary_info(model)

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
        """Constructs a token encoder.

        :param task:
            Must match :attr:`task`. If ``None``, defaults to :attr:`task`.
        :param lang:
            A language from :attr:`target_langs`. If ``None``, defaults to
            :attr:`default_target_lang`.
        :param mode:
            Must be 'target'. If ``None``, defaults to 'target'.
        :param device:
            The device on which to construct tensors.
        :param pin_memory:
            If ``True``, uses pinned memory while constructing tensors.
        """
        if task is not None and task != self._task:
            raise ValueError(f"`task` must be '{self._task}', but is '{task}' instead.")

        if mode is not None and mode != "target":
            raise ValueError(f"`mode` must be 'target', but is '{mode}' instead.")

        if lang is None:
            lang = self._default_target_lang

        if lang not in self._target_langs:
            raise NotSupportedError(
                f"`lang` must be a supported language, but is {lang} instead."
            )

        # For multilingual speech translation we prepend the language token.
        if self._task == "translation" and len(self._target_langs) > 1:
            prefix_tokens = ["</s>", f"<lang:{lang}>"]
        else:
            prefix_tokens = ["</s>"]

        return SentencePieceEncoder(
            self._model,
            prefix_tokens=prefix_tokens,
            device=device,
            pin_memory=pin_memory,
        )

    @override
    def create_raw_encoder(
        self, *, device: Device | None = None, pin_memory: bool = False
    ) -> TokenEncoder:
        return SentencePieceEncoder(self._model, device=device, pin_memory=pin_memory)

    @override
    def create_decoder(self, *, skip_special_tokens: bool = False) -> TokenDecoder:
        return SentencePieceDecoder(self._model)

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        return self._vocab_info


@dataclass
class S2TTransformerTokenizerConfig:
    task: Literal["translation", "transcription"] = "translation"

    target_langs: set[str] = field(default_factory=lambda: {"en"})

    default_target_lang: str = "en"


def load_s2t_transformer_tokenizer(
    path: Path, config: S2TTransformerTokenizerConfig
) -> Tokenizer:
    model = load_sentencepiece_model(path)

    return S2TTransformerTokenizer(
        model, config.task, config.target_langs, config.default_target_lang
    )
