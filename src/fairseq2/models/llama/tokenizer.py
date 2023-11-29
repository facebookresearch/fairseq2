# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import final

from fairseq2.data.text import BasicSentencePieceTokenizer
from fairseq2.data.typing import PathLike


@final
class LLaMATokenizer(BasicSentencePieceTokenizer):
    """Represents the tokenizer used by LLaMA models."""

    def __init__(self, pathname: PathLike) -> None:
        """
        :param pathname:
            The pathname of the SentencePiece model file.
        """
        super().__init__(pathname)
