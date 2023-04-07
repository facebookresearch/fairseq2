# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Set, final

import torch
from overrides import final as finaloverride

from fairseq2.data.text import (
    SentencePieceDecoder,
    SentencePieceEncoder,
    SentencePieceModel,
    TokenDecoder,
    TokenEncoder,
    Tokenizer,
)
from fairseq2.data.text.sentencepiece import vocab_from_sentencepiece
from fairseq2.data.typing import PathLike


@final
class S2TTransformerTokenizer(Tokenizer):
    """Represents the tokenizer used by S2T Transformer models."""

    model: SentencePieceModel
    task: str
    tgt_langs: Set[str]
    default_tgt_lang: str

    def __init__(
        self,
        pathname: PathLike,
        task: str,
        tgt_langs: Set[str],
        default_tgt_lang: str,
    ) -> None:
        """
        :param pathname:
            The pathname of the SentencePiece model file.
        :param task:
            The task for which to generate token indices. The valid values are
            'transcription' and 'translation'.
        :param tgt_langs:
            The list of supported target languages.
        :param default_tgt_lang:
            The fall-back language if no target language is specified.
        """
        if task != "transcription" and task != "translation":
            raise ValueError(
                f"`task` must be 'transcripton' or 'translation', but is '{task}' instead."
            )

        self.model = SentencePieceModel(pathname)

        self.task = task
        self.tgt_langs = tgt_langs
        self.default_tgt_lang = default_tgt_lang

        vocab_info = vocab_from_sentencepiece(self.model)

        super().__init__(vocab_info)

    @finaloverride
    def create_encoder(
        self,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        mode: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        pin_memory: bool = False,
        dtype: torch.dtype = torch.int32,
        disable_parallelism: bool = False,
    ) -> TokenEncoder:
        if task and task != self.task:
            raise ValueError(f"`task` must be '{self.task}', but is '{task}' instead.")

        if mode and mode != "target":
            raise ValueError(f"`mode` must be 'target', but is '{mode}' instead.")

        if not lang:
            lang = self.default_tgt_lang

        if lang not in self.tgt_langs:
            raise ValueError(
                f"`lang` must be a supported language, but is '{lang}' instead."
            )

        # For multilingual speech translation we prepend the language token.
        if self.task == "translation" and len(self.tgt_langs) > 1:
            prefix_tokens = ["</s>", f"<lang:{lang}>"]
        else:
            prefix_tokens = ["</s>"]

        return SentencePieceEncoder(
            self.model,
            prefix_tokens=prefix_tokens,
            batch_size=batch_size,
            device=device,
            pin_memory=pin_memory,
            dtype=dtype,
            disable_parallelism=disable_parallelism,
        )

    @finaloverride
    def create_decoder(self) -> TokenDecoder:
        return SentencePieceDecoder(self.model)
