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
    """Represents the tokenizer used by NLLB models."""

    model: SentencePieceModel

    def __init__(self, pathname: PathLike) -> None:
        """
        :param pathname:
            The pathname of the SentencePiece model file.
        """
        self.model = SentencePieceModel(pathname)

        vocabulary_info = vocabulary_from_sentencepiece(self.model)

        super().__init__(vocabulary_info)

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
            Not used in LLaMA, defaults to ``None``.
        :param lang:
            Not used in LLaMA, defaults to ``None``.
        :param mode:
            Not used in LLaMA, defaults to ``None``.
        :param device:
            The device on which to construct tensors.
        :param pin_memory:
            If ``True``, uses pinned memory while constructing tensors.
        """
        return SentencePieceEncoder(
            self.model,
            device=device,
            pin_memory=pin_memory,
        )

    @finaloverride
    def create_decoder(self) -> TextTokenDecoder:
        return SentencePieceDecoder(self.model)
