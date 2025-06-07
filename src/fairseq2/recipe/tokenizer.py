# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from fairseq2.assets import (
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetCardNotFoundError,
    AssetStore,
)
from fairseq2.data.tokenizers import (
    Tokenizer,
    TokenizerHandler,
    TokenizerLoadError,
    UnknownTokenizerError,
    UnknownTokenizerFamilyError,
    resolve_tokenizer_reference,
    tokenizer_asset_card_error,
)
from fairseq2.dependency import DependencyResolver
from fairseq2.error import SetupError
from fairseq2.logging import log
from fairseq2.recipe.config import TokenizerSection, get_config_section
from fairseq2.recipe.utils.log import log_tokenizer
from fairseq2.typing import Provider


def load_tokenizer(resolver: DependencyResolver) -> Tokenizer:
    return _do_load_tokenizer(resolver, section_name="tokenizer")


def load_source_tokenizer(resolver: DependencyResolver) -> Tokenizer:
    return _do_load_tokenizer(resolver, section_name="source_tokenizer")


def load_target_tokenizer(resolver: DependencyResolver) -> Tokenizer:
    return _do_load_tokenizer(resolver, section_name="target_tokenizer")


def _do_load_tokenizer(resolver: DependencyResolver, section_name: str) -> Tokenizer:
    tokenizer_section = get_config_section(resolver, section_name, TokenizerSection)

    asset_store = resolver.resolve(AssetStore)

    handlers = resolver.resolve_provider(TokenizerHandler)

    loader = _TokenizerLoader(asset_store, handlers)

    try:
        return loader.load(resolver, tokenizer_section)
    except TokenizerLoadError as ex:
        raise SetupError(
            f"The '{ex.tokenizer_name}' tokenizer cannot be loaded. See the nested exception for details."
        ) from ex


@final
class _TokenizerLoader:
    _asset_store: AssetStore
    _handlers: Provider[TokenizerHandler]

    def __init__(
        self, asset_store: AssetStore, handlers: Provider[TokenizerHandler]
    ) -> None:
        self._asset_store = asset_store
        self._handlers = handlers

    def load(
        self, resolver: DependencyResolver, tokenizer_section: TokenizerSection
    ) -> Tokenizer:
        name = tokenizer_section.name

        try:
            card = self._asset_store.retrieve_card(name)
        except AssetCardNotFoundError:
            raise UnknownTokenizerError(name) from None
        except AssetCardError as ex:
            raise tokenizer_asset_card_error(name) from ex

        try:
            card = resolve_tokenizer_reference(self._asset_store, card)
        except AssetCardError as ex:
            raise tokenizer_asset_card_error(name) from ex

        try:
            family = card.field("tokenizer_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise UnknownTokenizerError(name) from None
        except AssetCardError as ex:
            raise tokenizer_asset_card_error(name) from ex

        try:
            handler = self._handlers.resolve(family)
        except LookupError:
            raise UnknownTokenizerFamilyError(family, name) from None

        log.info("Loading '{}' tokenizer.", name)

        tokenizer = handler.load(resolver, card)

        log.info("Tokenizer loaded.")

        log_tokenizer(log, tokenizer)

        return tokenizer
