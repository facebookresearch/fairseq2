# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.data.tokenizers import Tokenizer
from fairseq2.recipe.composition.config import register_config_section
from fairseq2.recipe.config import TokenizerSection
from fairseq2.recipe.internal.tokenizer import _TokenizerHolder, _TokenizerLoader
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


def register_tokenizer(container: DependencyContainer, section_name: str) -> None:
    register_config_section(container, section_name, TokenizerSection, keyed=True)

    def load_tokenizer(resolver: DependencyResolver) -> _TokenizerHolder:
        section = resolver.resolve(TokenizerSection, key=section_name)

        tokenizer_loader = resolver.resolve(_TokenizerLoader)

        return tokenizer_loader.load(section_name, section)

    container.register(
        _TokenizerHolder, load_tokenizer, key=section_name, singleton=True
    )

    def get_tokenizer(resolver: DependencyResolver) -> Tokenizer:
        tokenizer_holder = resolver.resolve(_TokenizerHolder, key=section_name)

        return tokenizer_holder.tokenizer

    container.register(Tokenizer, get_tokenizer, key=section_name, singleton=True)


def _register_default_tokenizer(container: DependencyContainer) -> None:
    register_tokenizer(container, section_name="tokenizer")

    container.register_type(_TokenizerLoader)

    # Default Tokenizer
    def get_tokenizer_holder(resolver: DependencyResolver) -> _TokenizerHolder:
        return resolver.resolve(_TokenizerHolder, key="tokenizer")

    container.register(_TokenizerHolder, get_tokenizer_holder, singleton=True)

    def get_tokenizer(resolver: DependencyResolver) -> Tokenizer:
        return resolver.resolve(Tokenizer, key="tokenizer")

    container.register(Tokenizer, get_tokenizer, singleton=True)
