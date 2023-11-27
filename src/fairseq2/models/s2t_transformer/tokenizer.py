# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Set, final

from fairseq2.data.text import SentencePieceEncoder, SentencePieceTokenizerBase
from fairseq2.data.typing import PathLike
from fairseq2.typing import Device, finaloverride


@final
class S2TTransformerTokenizer(SentencePieceTokenizerBase):
    """Represents the tokenizer used by S2T Transformer models."""

    task: str
    target_langs: Set[str]
    default_target_lang: str

    def __init__(
        self,
        pathname: PathLike,
        task: str,
        target_langs: Set[str],
        default_target_lang: str,
    ) -> None:
        """
        :param pathname:
            The pathname of the SentencePiece model file.
        :param task:
            The task for which to generate token indices. The valid values are
            'transcription' and 'translation'.
        :param target_langs:
            The list of supported target languages.
        :param default_target_lang:
            The fall-back language if no target language is specified.
        """
        super().__init__(pathname)

        if task != "transcription" and task != "translation":
            raise ValueError(
                f"`task` must be 'transcripton' or 'translation', but is '{task}' instead."
            )

        self.task = task
        self.target_langs = target_langs
        self.default_target_lang = default_target_lang

    @finaloverride
    def create_encoder(
        self,
        *,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        mode: Optional[str] = None,
        device: Optional[Device] = None,
        pin_memory: bool = False,
    ) -> SentencePieceEncoder:
        """Create a token encoder.

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
        if task is not None and task != self.task:
            raise ValueError(f"`task` must be '{self.task}', but is '{task}' instead.")

        if mode is not None and mode != "target":
            raise ValueError(f"`mode` must be 'target', but is '{mode}' instead.")

        if lang is None:
            lang = self.default_target_lang

        if lang not in self.target_langs:
            raise ValueError(
                f"`lang` must be a supported language, but is '{lang}' instead."
            )

        # For multilingual speech translation we prepend the language token.
        if self.task == "translation" and len(self.target_langs) > 1:
            prefix_tokens = ["</s>", f"<lang:{lang}>"]
        else:
            prefix_tokens = ["</s>"]

        return SentencePieceEncoder(
            self.model,
            prefix_tokens=prefix_tokens,
            device=device,
            pin_memory=pin_memory,
        )
