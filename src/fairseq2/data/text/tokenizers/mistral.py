# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Final

from fairseq2.data.text.tokenizers.sentencepiece import (
    load_basic_sentencepiece_tokenizer,
)

MISTRAL_TOKENIZER_FAMILY: Final = "mistral"


load_mistral_tokenizer = load_basic_sentencepiece_tokenizer
