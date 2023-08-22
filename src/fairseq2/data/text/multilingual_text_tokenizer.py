# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Set, final

from fairseq2.data.text.sentencepiece import (
    SentencePieceDecoder,
    SentencePieceEncoder,
    SentencePieceModel,
    vocabulary_from_sentencepiece,
)
from fairseq2.data.text.text_tokenizer import (
    TextTokenDecoder,
    TextTokenEncoder,
    TextTokenizer,
)
from fairseq2.data.typing import PathLike
from fairseq2.typing import Device, finaloverride


@final
class MultilingualTextTokenizer(TextTokenizer):
    """Represents a generic bilingual/multilingual text tokenizer."""

    model: SentencePieceModel
    task: str
    source_langs: Set[str]
    target_langs: Set[str]
    default_source_lang: str
    default_target_lang: str

    def __init__(
        self,
        pathname: PathLike,
        task: str,
        source_langs: Set[str],
        target_langs: Set[str],
        default_source_lang: str,
        default_target_lang: str,
    ) -> None:
        """
        :param pathname:
            The pathname of the SentencePiece model file.
        :param task:
            A user-defined task. It is not used by the tokenizer, but will be
            validated in :meth:`create_encoder`.
        :param source_langs:
            The list of supported source languages.
        :param target_langs:
            The list of supported target languages.
        :param default_source_lang:
            The fall-back language if no source language is specified.
        :param default_target_lang:
            The fall-back language if no target language is specified.
        """
        self.model = SentencePieceModel(pathname)

        self.task = task

        self.source_langs = set(source_langs)
        self.target_langs = set(target_langs)

        self.default_source_lang = default_source_lang
        self.default_target_lang = default_target_lang

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
            Must match :attr:`task`. If ``None``, defaults to :attr:`task`.
        :param lang:
            A language from :attr:`source_langs` if ``mode`` is 'source', or a
            language from :attr:`target_langs` if ``mode`` is 'target'. If
            ``None``, defaults to either :attr:`default_source_lang` or
            :attr:`default_target_lang` depending on ``mode``.
        :param mode:
            Must be 'source' or 'target'. Set to 'source' if ``lang`` is the
            source language; set to 'target' if ``lang`` is the target language.
            If ``None``, defaults to 'source'.
        :param device:
            The device on which to construct tensors.
        :param pin_memory:
            If ``True``, uses pinned memory while constructing tensors.
        """
        if task is not None and task != self.task:
            raise ValueError(f"`task` must be '{self.task}', but is '{task}' instead.")

        if mode is None or mode == "source":
            if lang is None:
                lang = self.default_source_lang

            if lang not in self.source_langs:
                raise ValueError(
                    f"`lang` ({lang}) is not a supported source language. It must be one of: {self.source_langs}"
                )
        elif mode == "target":
            if lang is None:
                lang = self.default_target_lang

            if lang not in self.target_langs:
                raise ValueError(
                    f"`lang` ({lang}) is not a supported target language. It must be one of: {self.target_langs}"
                )
        else:
            raise ValueError(
                f"`mode` must be 'source' or 'target', but is '{mode}' instead."
            )

        return SentencePieceEncoder(
            self.model,
            prefix_tokens=[f"<lang:{lang}>"],
            suffix_tokens=["</s>"],
            device=device,
            pin_memory=pin_memory,
        )

    @finaloverride
    def create_decoder(self) -> TextTokenDecoder:
        return SentencePieceDecoder(self.model)
