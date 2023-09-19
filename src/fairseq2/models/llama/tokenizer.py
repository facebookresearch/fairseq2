# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, final

from overrides import final as finaloverride

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
from fairseq2.typing import Device


@final
class LLaMATokenizer(TextTokenizer):
    """Represents the tokenizer used by LLaMA models."""

    model: SentencePieceModel

    def __init__(self, pathname: PathLike) -> None:
        """
        :param pathname:
            The pathname of the SentencePiece model file.
        """
        self.model = SentencePieceModel(pathname)

        vocabulary_info = vocabulary_from_sentencepiece(self.model)

        # LLaMA tokenizer has no PAD symbol defined in its SentencePiece model
        # and uses EOS instead.
        vocabulary_info.pad_idx = vocabulary_info.eos_idx

        super().__init__(vocabulary_info)

    @finaloverride
    def create_encoder(
        self,
        *,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        mode: Optional[str] = None,
        device: Optional[Device] = None,
        pin_memory: bool = False,
    ) -> TextTokenEncoder:
        """Create a token encoder.

        :param task:
            Not used.
        :param lang:
            Not used.
        :param mode:
            Must be 'default' or 'prompt'. If ``None``, defaults to 'default'.
        :param device:
            The device on which to construct tensors.
        :param pin_memory:
            If ``True``, uses pinned memory while constructing tensors.
        """
        if task is not None:
            raise ValueError(f"`task` must be `None`, but is '{task}' instead.")

        if lang is not None:
            raise ValueError(f"`lang` must be `None`, but is '{lang}' instead.")

        if mode is None or mode == "default":
            prefix_tokens = ["<s>"]
            suffix_tokens = ["</s>"]
        elif mode == "prompt":
            prefix_tokens = ["<s>"]
            # In prompt mode, we expect the generator to finish the sequence.
            suffix_tokens = None
        else:
            raise ValueError(
                f"`mode` must be 'default' or 'prompt', but is '{mode}' instead."
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
