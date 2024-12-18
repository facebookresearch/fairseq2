# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

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
class S2TTransformerTokenizer(SentencePieceTokenizer):
    """Represents an S2T Transformer tokenizer."""

    _task: str
    _target_langs: set[str]
    _default_target_lang: str

    def __init__(
        self,
        path: Path,
        task: str,
        target_langs: set[str],
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
        super().__init__(path)

        if task != "transcription" and task != "translation":
            raise ValueError(
                f"`task` must be 'transcripton' or 'translation', but is '{task}' instead."
            )

        self._task = task
        self._target_langs = target_langs
        self._default_target_lang = default_target_lang

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
            raise ValueError(
                f"`lang` must be a supported language, but is '{lang}' instead."
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


S2T_TRANSFORMER_TOKENIZER_FAMILY: Final = "s2t_transformer"


def load_s2t_transformer_tokenizer(
    path: Path, card: AssetCard
) -> S2TTransformerTokenizer:
    valid_tasks = {"translation", "transcription"}

    task = card.field("task").as_one_of(valid_tasks)

    target_langs = card.field("target_langs").as_(list[str])

    try:
        return S2TTransformerTokenizer(
            path, task, set(target_langs), default_target_lang=target_langs[0]
        )
    except ValueError as ex:
        raise AssetCardError(
            f"The '{card.name}' asset card does not have a valid text tokenizer configuration. See the nested exception for details."
        ) from ex
