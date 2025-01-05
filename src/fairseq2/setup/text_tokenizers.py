# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.data.text import (
    CHAR_TOKENIZER_FAMILY,
    LLAMA_TOKENIZER_FAMILY,
    MISTRAL_TOKENIZER_FAMILY,
    NLLB_TOKENIZER_FAMILY,
    S2T_TRANSFORMER_TOKENIZER_FAMILY,
    StandardTextTokenizerHandler,
    TextTokenizerHandler,
    TextTokenizerLoader,
)
from fairseq2.data.text.tokenizers.char_tokenizer import load_char_tokenizer
from fairseq2.data.text.tokenizers.llama import load_llama_tokenizer
from fairseq2.data.text.tokenizers.mistral import load_mistral_tokenizer
from fairseq2.data.text.tokenizers.nllb import load_nllb_tokenizer
from fairseq2.data.text.tokenizers.s2t_transformer import load_s2t_transformer_tokenizer


def _register_text_tokenizers(context: RuntimeContext) -> None:
    register_text_tokenizer(
        context,
        CHAR_TOKENIZER_FAMILY,
        loader=load_char_tokenizer,
    )

    register_text_tokenizer(
        context,
        LLAMA_TOKENIZER_FAMILY,
        loader=load_llama_tokenizer,
    )

    register_text_tokenizer(
        context,
        MISTRAL_TOKENIZER_FAMILY,
        loader=load_mistral_tokenizer,
    )

    register_text_tokenizer(
        context,
        NLLB_TOKENIZER_FAMILY,
        loader=load_nllb_tokenizer,
    )

    register_text_tokenizer(
        context,
        S2T_TRANSFORMER_TOKENIZER_FAMILY,
        loader=load_s2t_transformer_tokenizer,
    )


def register_text_tokenizer(
    context: RuntimeContext, family: str, *, loader: TextTokenizerLoader
) -> None:
    handler = StandardTextTokenizerHandler(
        loader=loader, asset_download_manager=context.asset_download_manager
    )

    registry = context.get_registry(TextTokenizerHandler)

    registry.register(family, handler)
