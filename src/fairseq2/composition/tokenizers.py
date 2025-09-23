# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from types import NoneType
from typing import Protocol, TypeVar

from fairseq2.data.tokenizers import (
    StandardTokenizerFamily,
    Tokenizer,
    TokenizerFamily,
    TokenizerLoader,
)
from fairseq2.data.tokenizers.char import CHAR_TOKENIZER_FAMILY, load_char_tokenizer
from fairseq2.error import InternalError
from fairseq2.models.llama import (
    LLAMA_FAMILY,
    LLaMATokenizerConfig,
    load_llama_tokenizer,
)
from fairseq2.models.llama4 import (
    LLAMA4_FAMILY,
    Llama4TokenizerConfig,
    load_llama4_tokenizer,
)
from fairseq2.models.mistral import MISTRAL_FAMILY, load_mistral_tokenizer
from fairseq2.models.nllb import (
    NLLB_FAMILY,
    NllbTokenizer,
    NllbTokenizerConfig,
    load_nllb_tokenizer,
)
from fairseq2.models.qwen import (
    QWEN_FAMILY,
    QwenTokenizer,
    QwenTokenizerConfig,
    load_qwen_tokenizer,
)
from fairseq2.models.s2t_transformer import (
    S2T_TRANSFORMER_FAMILY,
    S2TTransformerTokenizer,
    S2TTransformerTokenizerConfig,
    load_s2t_transformer_tokenizer,
)
from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyResolver,
    wire_object,
)

TokenizerConfigT_contra = TypeVar("TokenizerConfigT_contra", contravariant=True)


class AdvancedTokenizerLoader(Protocol[TokenizerConfigT_contra]):
    def __call__(
        self, resolver: DependencyResolver, path: Path, config: TokenizerConfigT_contra
    ) -> Tokenizer: ...


TokenizerT = TypeVar("TokenizerT", bound=Tokenizer)

TokenizerConfigT = TypeVar("TokenizerConfigT")


def register_tokenizer_family(
    container: DependencyContainer,
    name: str,
    kls: type[TokenizerT],
    config_kls: type[TokenizerConfigT],
    *,
    loader: TokenizerLoader[TokenizerConfigT] | None = None,
    advanced_loader: AdvancedTokenizerLoader[TokenizerConfigT] | None = None,
) -> None:
    if advanced_loader is not None:
        if loader is not None:
            raise ValueError(
                "`loader` and `advanced_loader` must not be specified at the same time."
            )
    elif loader is None:
        raise ValueError("`loader` or `advanced_loader` must be specified.")

    def create_family(resolver: DependencyResolver) -> TokenizerFamily:
        nonlocal loader

        if advanced_loader is not None:

            def load_tokenizer(path: Path, config: TokenizerConfigT) -> Tokenizer:
                return advanced_loader(resolver, path, config)

            loader = load_tokenizer
        elif loader is None:
            raise InternalError("`loader` is `None`.")

        return wire_object(
            resolver,
            StandardTokenizerFamily,
            name=name,
            kls=kls,
            config_kls=config_kls,
            loader=loader,
        )

    container.register(TokenizerFamily, create_family, key=name)


def _register_tokenizer_families(container: DependencyContainer) -> None:
    # Char
    register_tokenizer_family(
        container,
        CHAR_TOKENIZER_FAMILY,
        kls=Tokenizer,
        config_kls=NoneType,
        loader=load_char_tokenizer,
    )

    # LLaMA
    register_tokenizer_family(
        container,
        LLAMA_FAMILY,
        kls=Tokenizer,
        config_kls=LLaMATokenizerConfig,
        loader=load_llama_tokenizer,
    )

    # Llama 4
    register_tokenizer_family(
        container,
        LLAMA4_FAMILY,
        kls=Tokenizer,
        config_kls=Llama4TokenizerConfig,
        loader=load_llama4_tokenizer,
    )

    # Mistral
    register_tokenizer_family(
        container,
        MISTRAL_FAMILY,
        kls=Tokenizer,
        config_kls=NoneType,
        loader=load_mistral_tokenizer,
    )

    # Qwen
    register_tokenizer_family(
        container,
        QWEN_FAMILY,
        kls=QwenTokenizer,
        config_kls=QwenTokenizerConfig,
        loader=load_qwen_tokenizer,
    )

    # NLLB
    register_tokenizer_family(
        container,
        NLLB_FAMILY,
        kls=NllbTokenizer,
        config_kls=NllbTokenizerConfig,
        loader=load_nllb_tokenizer,
    )

    # S2T Transformer
    register_tokenizer_family(
        container,
        S2T_TRANSFORMER_FAMILY,
        kls=S2TTransformerTokenizer,
        config_kls=S2TTransformerTokenizerConfig,
        loader=load_s2t_transformer_tokenizer,
    )
