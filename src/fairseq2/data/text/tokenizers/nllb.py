# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final, final

from typing_extensions import override

from fairseq2.assets import AssetCard, AssetCardError
from fairseq2.data.text.tokenizers.sentencepiece import (
    SentencePieceEncoder,
    SentencePieceTokenizer,
)
from fairseq2.typing import Device


@final
class NllbTokenizer(SentencePieceTokenizer):
    """Represents an NLLB tokenizer."""

    _langs: set[str]
    _default_lang: str

    def __init__(self, path: Path, langs: Sequence[str], default_lang: str) -> None:
        """
        :param path:
            The path to the SentencePiece model file.
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
        # PAD symbol. We use an undocumented feature of our C++ API to insert
        # it to the model at index 0.
        control_symbols.append("<pad>@0")

        super().__init__(path, control_symbols)

        self._langs = set(langs)

        self._default_lang = default_lang

    @override
    def create_encoder(
        self,
        *,
        task: str | None = None,
        lang: str | None = None,
        mode: str | None = None,
        device: Device | None = None,
        pin_memory: bool = False,
    ) -> SentencePieceEncoder:
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


NLLB_TOKENIZER_FAMILY: Final = "nllb"


def load_nllb_tokenizer(path: Path, card: AssetCard) -> NllbTokenizer:
    langs = card.field("langs").as_(list[str])

    default_lang = card.field("default_lang").as_(str)

    try:
        return NllbTokenizer(path, langs, default_lang)
    except ValueError as ex:
        raise AssetCardError(
            f"The '{card.name}' asset card does not have a valid text tokenizer configuration. See the nested exception for details."
        ) from ex
