# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Set, final

import torch
from overrides import final as finaloverride

from fairseq2.data.text.sentencepiece import (
    SentencePieceDecoder,
    SentencePieceEncoder,
    SentencePieceModel,
    vocab_from_sentencepiece,
)
from fairseq2.data.text.tokenizer import TokenDecoder, TokenEncoder, Tokenizer
from fairseq2.data.typing import PathLike


@final
class MultilingualTokenizer(Tokenizer):
    """Represents a generic bilingual/multilingual tokenizer."""

    model: SentencePieceModel
    task: str
    src_langs: Set[str]
    tgt_langs: Set[str]
    default_src_lang: str
    default_tgt_lang: str

    def __init__(
        self,
        pathname: PathLike,
        task: str,
        src_langs: Set[str],
        tgt_langs: Set[str],
        default_src_lang: str,
        default_tgt_lang: str,
    ) -> None:
        """
        :param pathname:
            The pathname of the SentencePiece model file.
        :param task:
            A user-defined task; it has no meaning to the tokenizer, but will
            be checked in :meth:`create_encoder`.
        :param src_langs:
            The list of supported source languages.
        :param tgt_langs:
            The list of supported target languages.
        :param default_src_lang:
            The fall-back language if no source language is specified.
        :param default_tgt_lang:
            The fall-back language if no target language is specified.
        """
        self.model = SentencePieceModel(pathname)

        self.task = task

        self.src_langs = set(src_langs)
        self.tgt_langs = set(tgt_langs)

        self.default_src_lang = default_src_lang
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
        dtype: torch.dtype = torch.int64,
        disable_parallelism: bool = False,
    ) -> TokenEncoder:
        """Create a token encoder.

        :param task:
            The specified task must match :attr:`task`.
        :param lang:
            A language from :attr:`src_langs` if ``mode`` is 'source', or a
            language from :attr:`tgt_langs` if ``mode`` is 'target'. If
            ``None``, defaults to either :attr:`default_src_lang` or
            :attr:`default_tgt_lang` depending on the mode.
        :param mode:
            The valid values are 'source' and 'target'. Set to 'source' if
            ``lang`` is the source language; set to 'target' if ``lang`` is the
            target language. If ``None``, defaults to 'source'.
        :param batch_size:
            If the number of sentences to encode is less than ``batch_size``,
            the output will be padded.
        :param device:
            The device on which to initialize token indices.
        :param pin_memory:
            If ``True``, uses pinned memory before copying token indices to the
            target device. (only supported by CUDA devices)
        :param dtype:
            The integral data type of generated token indices.
        :param disabled_parallelism:
            If ``True``, disables parallelism and uses the calling thread only.
        """
        if task and task != self.task:
            raise ValueError(f"`task` must be '{self.task}', but is '{task}' instead.")

        if not mode or mode == "source":
            if not lang:
                lang = self.default_src_lang

            if lang not in self.src_langs:
                raise ValueError(
                    f"`lang` must be a supported source language, but is '{lang}' instead."
                )
        elif mode == "target":
            if not lang:
                lang = self.default_tgt_lang

            if lang not in self.tgt_langs:
                raise ValueError(
                    f"`lang` must be a supported target language, but is '{lang}' instead."
                )
        else:
            raise ValueError(
                f"`mode` must be 'source' or 'target', but is '{mode}' instead."
            )

        return SentencePieceEncoder(
            self.model,
            prefix_tokens=[f"<lang:{lang}>"],
            suffix_tokens=["</s>"],
            batch_size=batch_size,
            device=device,
            pin_memory=pin_memory,
            dtype=dtype,
            disable_parallelism=disable_parallelism,
        )

    @finaloverride
    def create_decoder(self) -> TokenDecoder:
        return SentencePieceDecoder(self.model)
