# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.data.tokenizers import Tokenizer
from fairseq2.recipe.composition.config import register_config_section
from fairseq2.recipe.config import TokenizerSection
from fairseq2.recipe.internal.tokenizer import _RecipeTokenizerLoader
from fairseq2.recipe.tokenizer import RecipeTokenizer
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


def register_tokenizer(container: DependencyContainer, section_name: str) -> None:
    register_config_section(container, section_name, TokenizerSection, keyed=True)

    def load_tokenizer(resolver: DependencyResolver) -> RecipeTokenizer:
        section = resolver.resolve(TokenizerSection, key=section_name)

        tokenizer_loader = resolver.resolve(_RecipeTokenizerLoader)

        return tokenizer_loader.load(section_name, section)

    container.register(
        RecipeTokenizer, load_tokenizer, key=section_name, singleton=True
    )

    if section_name == "tokenizer":

        def get_default_tokenizer(resolver: DependencyResolver) -> Tokenizer:
            return resolver.resolve(RecipeTokenizer, key=section_name)

        container.register(Tokenizer, get_default_tokenizer)


def _register_tokenizers(container: DependencyContainer) -> None:
    register_tokenizer(container, section_name="tokenizer")

    container.register_type(_RecipeTokenizerLoader)
