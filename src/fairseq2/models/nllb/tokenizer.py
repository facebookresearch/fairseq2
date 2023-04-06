# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence, Set, final

import torch
from overrides import final as finaloverride

from fairseq2.data.text import (
    SentencePieceDecoder,
    SentencePieceEncoder,
    SentencePieceModel,
    TokenDecoder,
    TokenEncoder,
    Tokenizer,
    vocab_from_sentencepiece,
)
from fairseq2.data.typing import PathLike


@final
class NllbTokenizer(Tokenizer):
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
        # Each language is represented by a __lang__ control token.
        control_tokens = [f"__{lang}__" for lang in langs]

        # Internal control tokens that are not relevant for public use.
        control_tokens.extend(["<MINED_DATA>", "<NMT_BT_DATA>", "<SMT_BT_DATA>"])

        # The SentencePiece model of NLLB is peculiar as it does not define a
        # pad token. We use an undocumented feature of our underlying C++ API
        # to insert a pad token to the model at index 0.
        control_tokens.append("<pad>@0")

        self.model = SentencePieceModel(pathname, control_tokens)

        self.langs = set(langs)

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
            The only valid value is 'translation'. If ``None``, defaults to
            'translation'.
        :param lang:
            A language from :attr:`langs`.
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
        if task and task != "translation":
            raise ValueError(f"`task` must be 'translation', but is '{task}' instead.")

        if not lang:
            lang = self.default_lang

        if lang not in self.langs:
            raise ValueError(
                f"`lang` must be a supported language, but is '{lang}' instead."
            )

        if not mode or mode == "source":
            # NLLB models expect a language token in place of BOS in source
            # sequences.
            prefix_tokens = [f"__{lang}__"]
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
            batch_size=batch_size,
            device=device,
            pin_memory=pin_memory,
            dtype=dtype,
            disable_parallelism=disable_parallelism,
        )

    @finaloverride
    def create_decoder(self) -> TokenDecoder:
        return SentencePieceDecoder(self.model)
