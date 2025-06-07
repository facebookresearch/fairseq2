# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from types import NoneType

from fairseq2.data.tokenizers import register_tokenizer_family
from fairseq2.data.tokenizers.char import CHAR_TOKENIZER_FAMILY, load_char_tokenizer
from fairseq2.models.llama import (
    LLAMA_FAMILY,
    LLaMATokenizerConfig,
    load_llama_tokenizer,
)
from fairseq2.models.mistral import MISTRAL_FAMILY, load_mistral_tokenizer
from fairseq2.models.nllb import NLLB_FAMILY, NllbTokenizerConfig, load_nllb_tokenizer
from fairseq2.models.qwen import QWEN_FAMILY, QwenTokenizerConfig, load_qwen_tokenizer
from fairseq2.models.s2t_transformer import (
    S2T_TRANSFORMER_FAMILY,
    S2TTransformerTokenizerConfig,
    load_s2t_transformer_tokenizer,
)
from fairseq2.runtime.dependency import DependencyContainer


def _register_tokenizer_families(container: DependencyContainer) -> None:
    # Char
    register_tokenizer_family(
        container,
        CHAR_TOKENIZER_FAMILY,
        config_kls=NoneType,
        loader=load_char_tokenizer,
    )

    # LLaMA
    register_tokenizer_family(
        container,
        LLAMA_FAMILY,
        config_kls=LLaMATokenizerConfig,
        loader=load_llama_tokenizer,
    )

    # Mistral
    register_tokenizer_family(
        container,
        MISTRAL_FAMILY,
        config_kls=NoneType,
        loader=load_mistral_tokenizer,
    )

    # Qwen
    register_tokenizer_family(
        container,
        QWEN_FAMILY,
        config_kls=QwenTokenizerConfig,
        loader=load_qwen_tokenizer,
    )

    # NLLB
    register_tokenizer_family(
        container,
        NLLB_FAMILY,
        config_kls=NllbTokenizerConfig,
        loader=load_nllb_tokenizer,
    )

    # S2T Transformer
    register_tokenizer_family(
        container,
        S2T_TRANSFORMER_FAMILY,
        config_kls=S2TTransformerTokenizerConfig,
        loader=load_s2t_transformer_tokenizer,
    )
