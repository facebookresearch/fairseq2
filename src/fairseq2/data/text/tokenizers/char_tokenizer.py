# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Final, final

from typing_extensions import override

from fairseq2.data.text.tokenizers.sentencepiece import RawSentencePieceTokenizerHandler

CHAR_TOKENIZER_FAMILY: Final = "char_tokenizer"


@final
class CharTokenizerHandler(RawSentencePieceTokenizerHandler):
    @property
    @override
    def family(self) -> str:
        return CHAR_TOKENIZER_FAMILY
