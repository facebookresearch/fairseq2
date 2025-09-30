# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Final

from fairseq2.data.tokenizers.sentencepiece import (
    RawSentencePieceTokenizer,
    load_sentencepiece_model,
)
from fairseq2.data.tokenizers.tokenizer import Tokenizer

CHAR_TOKENIZER_FAMILY: Final = "char_tokenizer"


def load_char_tokenizer(path: Path, config: None) -> Tokenizer:
    model = load_sentencepiece_model(path)

    return RawSentencePieceTokenizer(model)
