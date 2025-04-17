# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from fairseq2.context import RuntimeContext
from fairseq2.data.text.tokenizers import (
    StandardTextTokenizerHandler,
    TextTokenizerHandler,
    TextTokenizerLoader,
)
from fairseq2.data.text.tokenizers.char_tokenizer import (
    CHAR_TOKENIZER_FAMILY,
    load_char_tokenizer,
)
from fairseq2.data.text.tokenizers.llama import (
    LLAMA_TOKENIZER_FAMILY,
    load_llama_tokenizer,
)
from fairseq2.data.text.tokenizers.qwen import (
    QWEN_TOKENIZER_FAMILY,
    load_qwen_tokenizer,
)
from fairseq2.data.text.tokenizers.mistral import (
    MISTRAL_TOKENIZER_FAMILY,
    load_mistral_tokenizer,
)
from fairseq2.data.text.tokenizers.nllb import (
    NLLB_TOKENIZER_FAMILY,
    load_nllb_tokenizer,
)
from fairseq2.data.text.tokenizers.s2t_transformer import (
    S2T_TRANSFORMER_TOKENIZER_FAMILY,
    load_s2t_transformer_tokenizer,
)
from fairseq2.registry import Registry


def register_text_tokenizer_families(context: RuntimeContext) -> None:
    # fmt: off
    registrar = TextTokenizerRegistrar(context)

    # Char Tokenizer
    registrar.register_family(
        CHAR_TOKENIZER_FAMILY, load_char_tokenizer
    )

    # LLaMA
    registrar.register_family(
        LLAMA_TOKENIZER_FAMILY, load_llama_tokenizer
    )

    # Qwen
    registrar.register_family(
        QWEN_TOKENIZER_FAMILY, load_qwen_tokenizer
    )

    # NLLB
    registrar.register_family(
        NLLB_TOKENIZER_FAMILY, load_nllb_tokenizer
    )

    # Mistral
    registrar.register_family(
        MISTRAL_TOKENIZER_FAMILY, load_mistral_tokenizer
    )

    # S2T Transformer
    registrar.register_family(
        S2T_TRANSFORMER_TOKENIZER_FAMILY, load_s2t_transformer_tokenizer
    )

    # fmt: on


@final
class TextTokenizerRegistrar:
    _context: RuntimeContext
    _registry: Registry[TextTokenizerHandler]

    def __init__(self, context: RuntimeContext) -> None:
        self._context = context

        self._registry = context.get_registry(TextTokenizerHandler)

    def register_family(self, family: str, loader: TextTokenizerLoader) -> None:
        asset_download_manager = self._context.asset_download_manager

        handler = StandardTextTokenizerHandler(family, loader, asset_download_manager)

        self._registry.register(handler.family, handler)
