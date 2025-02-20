# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.data.text.tokenizers.char_tokenizer import register_char_tokenizer
from fairseq2.data.text.tokenizers.llama import register_llama_tokenizer
from fairseq2.data.text.tokenizers.mistral import register_mistral_tokenizer
from fairseq2.data.text.tokenizers.nllb import register_nllb_tokenizer
from fairseq2.data.text.tokenizers.s2t_transformer import (
    register_s2t_transformer_tokenizer,
)


def register_text_tokenizers(context: RuntimeContext) -> None:
    register_char_tokenizer(context)
    register_llama_tokenizer(context)
    register_mistral_tokenizer(context)
    register_nllb_tokenizer(context)
    register_s2t_transformer_tokenizer(context)
