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
class NllbTokenizer(TextTokenizer):
    """Represents the tokenizer used by NLLB models."""

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
        # Each language is represented by a `__lang__` control symbol.
        control_symbols = [f"__{lang}__" for lang in langs]

        # Internal control symbols that are not relevant for eval use.
        control_symbols.extend(["<MINED_DATA>", "<MMT_BT_DATA>", "<SMT_BT_DATA>"])

        # The SentencePiece model of NLLB is peculiar as it does not define a
        # pad symbol. We use an undocumented feature of our C++ API to insert
        # a pad symbol to the model at index 0.
        control_symbols.append("<pad>@0")

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
            # NLLB models expect a language token in place of BOS in source
            # sequences.
            prefix_tokens = [f"__{lang}__"]
            suffix_tokens = ["</s>"]
        elif mode == "source_mining":
            prefix_tokens = [f"__{lang}__", "<MINED_DATA>"]
            suffix_tokens = ["</s>"]
        elif mode == "source_mmt_bt":
            prefix_tokens = [f"__{lang}__", "<MMT_BT_DATA>"]
            suffix_tokens = ["</s>"]
        elif mode == "source_smt_bt":
            prefix_tokens = [f"__{lang}__", "<SMT_BT_DATA>"]
            suffix_tokens = ["</s>"]
        elif mode == "target":
            # Target sequences are expected to start with an EOS, followed by
            # the language token.
            prefix_tokens = ["</s>", f"__{lang}__"]
            suffix_tokens = []
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
