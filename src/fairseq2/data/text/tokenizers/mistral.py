# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Final, final

from typing_extensions import override

from fairseq2.data.text.tokenizers.sentencepiece import (
    BasicSentencePieceTokenizerHandler,
)

MISTRAL_TOKENIZER_FAMILY: Final = "mistral"


@final
class MistralTokenizerHandler(BasicSentencePieceTokenizerHandler):
    @property
    @override
    def family(self) -> str:
        return MISTRAL_TOKENIZER_FAMILY
