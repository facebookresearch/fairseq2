# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.data.text.tokenizers.char_tokenizer import (
    CHAR_TOKENIZER_FAMILY,
    load_char_tokenizer,
)
from fairseq2.data.text.tokenizers.llama import (
    LLAMA_TOKENIZER_FAMILY,
    load_llama_tokenizer,
)
from fairseq2.data.text.tokenizers.mistral import (
    MISTRAL_TOKENIZER_FAMILY,
    load_mistral_tokenizer,
)
from fairseq2.data.text.tokenizers.nllb import (
    NLLB_TOKENIZER_FAMILY,
    load_nllb_tokenizer,
)
from fairseq2.data.text.tokenizers.registry import (
    StandardTextTokenizerHandler,
    TextTokenizerRegistry,
)
from fairseq2.data.text.tokenizers.s2t_transformer import (
    S2T_TRANSFORMER_TOKENIZER_FAMILY,
    load_s2t_transformer_tokenizer,
)
from fairseq2.extensions import run_extensions


def register_text_tokenizers(registry: TextTokenizerRegistry) -> None:
    # Char Tokenizer
    handler = StandardTextTokenizerHandler(loader=load_char_tokenizer)

    registry.register(CHAR_TOKENIZER_FAMILY, handler)

    # LLaMA
    handler = StandardTextTokenizerHandler(loader=load_llama_tokenizer)

    registry.register(LLAMA_TOKENIZER_FAMILY, handler)

    # Mistral
    handler = StandardTextTokenizerHandler(loader=load_mistral_tokenizer)

    registry.register(MISTRAL_TOKENIZER_FAMILY, handler)

    # NLLB
    handler = StandardTextTokenizerHandler(loader=load_nllb_tokenizer)

    registry.register(NLLB_TOKENIZER_FAMILY, handler)

    # S2T Transformer
    handler = StandardTextTokenizerHandler(loader=load_s2t_transformer_tokenizer)

    registry.register(S2T_TRANSFORMER_TOKENIZER_FAMILY, handler)

    # Extensions
    run_extensions("register_fairseq2_text_tokenizers", registry)
