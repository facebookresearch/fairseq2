# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

from fairseq2.data.tokenizers import Tokenizer
from fairseq2.data.tokenizers.sentencepiece import (
    BasicSentencePieceTokenizer,
    load_sentencepiece_model,
)


def load_mistral_tokenizer(path: Path, config: None) -> Tokenizer:
    model = load_sentencepiece_model(path)

    return BasicSentencePieceTokenizer(model)
