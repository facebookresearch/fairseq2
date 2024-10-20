# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence, Set, final

from fairseq2.data.text import (
    SentencePieceDecoder,
    SentencePieceEncoder,
    SentencePieceModel,
    TextTokenDecoder,
    TextTokenEncoder,
    TextTokenizer,
    vocabulary_from_sentencepiece,
)
from fairseq2.data.typing import PathLike
from fairseq2.typing import Device, finaloverride


@final
class mBartTokenizer(TextTokenizer):
    """Represents the tokenizer used by mBart models."""

    model: SentencePieceModel
    langs: Set[str]
    default_lang: str

    def __init__(
        self, pathname: PathLike, langs: Sequence[str], default_lang: str
    ) -> None:
        """
        :param pathname:
            The pathname of the SentencePiece model file.
        :param langs:
            The list of supported languages.
        :param default_lang:
            The fall-back language if no language is specified.
        """
        # Each language is represented by a `[lang]` control symbol.
        control_symbols = [f"[{lang}]" for lang in langs]

        control_symbols.append("<mask>")

        self.model = SentencePieceModel(pathname, control_symbols)

        self.langs = set(langs)

        self.default_lang = default_lang

        vocab_info = vocabulary_from_sentencepiece(self.model)

        super().__init__(vocab_info)

    @finaloverride
    def create_encoder(
        self,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        mode: Optional[str] = None,
        device: Optional[Device] = None,
        pin_memory: bool = False,
    ) -> TextTokenEncoder:
        """Create a token encoder.

        :param task:
            Must be 'translation'. If ``None``, defaults to 'translation'.
        :param lang:
            A language from :attr:`langs`. If ``None``, defaults to
            :attr:`default_lang`.
        :param mode:
            Must be 'source' or 'target'. Set to 'source' if ``lang`` is the
            source language; set to 'target' if ``lang`` is the target language.
            If ``None``, defaults to 'source'.
        :param device:
            The device on which to construct tensors.
        :param pin_memory:
            If ``True``, uses pinned memory while constructing tensors.
        """
        if task is not None and task != "translation":
            raise ValueError(f"`task` must be 'translation', but is '{task}' instead.")

        if lang is None:
            lang = self.default_lang

        if lang not in self.langs:
            raise ValueError(
                f"`lang` must be a supported language, but is '{lang}' instead."
            )

        if mode is None or mode == "source":
            prefix_tokens = ["<s>"]
            suffix_tokens = ["</s>", f"[{lang}]"]
        elif mode == "target":
            prefix_tokens = [f"[{lang}]", "<s>"]
            suffix_tokens = ["</s>"]
        else:
            raise ValueError(
                f"`mode` must be 'source' or 'target', but is '{mode}' instead."
            )

        return SentencePieceEncoder(
            self.model,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            device=device,
            pin_memory=pin_memory,
        )

    @finaloverride
    def create_decoder(self) -> TextTokenDecoder:
        return SentencePieceDecoder(self.model)
