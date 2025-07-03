# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Set
from pathlib import Path
from typing import Final, final

from typing_extensions import override

from fairseq2.assets import AssetCard, AssetCardError
from fairseq2.data import VocabularyInfo
from fairseq2.data.text.tokenizers import (
    TextTokenDecoder,
    TextTokenEncoder,
    TextTokenizer,
    TextTokenizerLoadError,
    text_tokenizer_asset_card_error,
)
from fairseq2.data.text.tokenizers.sentencepiece import (
    SentencePieceDecoder,
    SentencePieceEncoder,
    SentencePieceModel,
    get_sentencepiece_vocabulary_info,
)
from fairseq2.device import Device


@final
class NllbTokenizer(TextTokenizer):
    """Represents an NLLB tokenizer."""

    _model: SentencePieceModel
    _langs: Set[str]
    _default_lang: str
    _vocab_info: VocabularyInfo

    def __init__(
        self, model: SentencePieceModel, langs: Set[str], default_lang: str
    ) -> None:
        """
        :param langs: The list of supported languages.
        :param default_lang: The fall-back language if no language is specified.
        """
        self._model = model

        self._langs = langs

        self._default_lang = default_lang

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
    ) -> TextTokenEncoder:
        """Constructs a token encoder.

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
            lang = self._default_lang

        if lang not in self._langs:
            raise ValueError(
                f"`lang` must be a supported language, but is '{lang}' instead."
            )

        match mode:
            case None | "source":
                # NLLB models expect a language token in place of BOS in source
                # sequences.
                prefix_tokens = [f"__{lang}__"]
                suffix_tokens = ["</s>"]
            case "source_mining":
                prefix_tokens = [f"__{lang}__", "<MINED_DATA>"]
                suffix_tokens = ["</s>"]
            case "source_mmt_bt":
                prefix_tokens = [f"__{lang}__", "<MMT_BT_DATA>"]
                suffix_tokens = ["</s>"]
            case "source_smt_bt":
                prefix_tokens = [f"__{lang}__", "<SMT_BT_DATA>"]
                suffix_tokens = ["</s>"]
            case "target":
                # Target sequences are expected to start with an EOS, followed by
                # the language token.
                prefix_tokens = ["</s>", f"__{lang}__"]
                suffix_tokens = ["</s>"]
            case _:
                raise ValueError(
                    f"`mode` must be 'source' or 'target', but is '{mode}' instead."
                )

        return SentencePieceEncoder(
            self._model,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            device=device,
            pin_memory=pin_memory,
        )

    @override
    def create_raw_encoder(
        self, *, device: Device | None = None, pin_memory: bool = False
    ) -> TextTokenEncoder:
        return SentencePieceEncoder(self._model, device=device, pin_memory=pin_memory)

    @override
    def create_decoder(self, *, skip_special_tokens: bool = False) -> TextTokenDecoder:
        return SentencePieceDecoder(self._model)

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        return self._vocab_info


NLLB_TOKENIZER_FAMILY: Final = "nllb"


def load_nllb_tokenizer(path: Path, card: AssetCard) -> TextTokenizer:
    try:
        langs = card.field("langs").as_(list[str])
    except AssetCardError as ex:
        raise text_tokenizer_asset_card_error(card.name) from ex

    # Each language is represented by a `__lang__` control symbol.
    control_symbols = [f"__{lang}__" for lang in langs]

    # Internal control symbols that are not relevant for eval use.
    control_symbols.extend(["<MINED_DATA>", "<MMT_BT_DATA>", "<SMT_BT_DATA>"])

    # The SentencePiece model of NLLB is peculiar as it does not define a
    # PAD symbol. We use an undocumented feature of our C++ API to insert
    # it to the model at index 0.
    control_symbols.append("<pad>@0")

    try:
        model = SentencePieceModel(path, control_symbols)
    except (OSError, RuntimeError) as ex:
        raise TextTokenizerLoadError(
            card.name, f"The '{card.name}' text tokenizer model cannot be loaded. See the nested exception for details."  # fmt: skip
        ) from ex

    try:
        default_lang = card.field("default_lang").as_(str)
    except AssetCardError as ex:
        raise text_tokenizer_asset_card_error(card.name) from ex

    return NllbTokenizer(model, set(langs), default_lang)
