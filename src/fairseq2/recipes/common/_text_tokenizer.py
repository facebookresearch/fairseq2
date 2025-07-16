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
from fairseq2.context import RuntimeContext
from fairseq2.data.text.tokenizers import (
    TextTokenizer,
    TextTokenizerHandler,
    TextTokenizerLoadError,
    UnknownTextTokenizerError,
    UnknownTextTokenizerFamilyError,
    resolve_text_tokenizer_reference,
    text_tokenizer_asset_card_error,
)
from fairseq2.logging import log
from fairseq2.recipes import RecipeError
from fairseq2.recipes.config import TextTokenizerSection
from fairseq2.recipes.utils.log import log_tokenizer
from fairseq2.registry import Provider


def load_text_tokenizer(
    context: RuntimeContext, tokenizer_section: TextTokenizerSection
) -> TextTokenizer:
    tokenizer_handlers = context.get_registry(TextTokenizerHandler)

    loader = _TextTokenizerLoader(context.asset_store, tokenizer_handlers)

    try:
        return loader.load(tokenizer_section)
    except TextTokenizerLoadError as ex:
        raise RecipeError(
            f"The '{ex.tokenizer_name}' tokenizer cannot be loaded. See the nested exception for details."
        ) from ex


@final
class _TextTokenizerLoader:
    _asset_store: AssetStore
    _tokenizer_handlers: Provider[TextTokenizerHandler]

    def __init__(
        self,
        asset_store: AssetStore,
        tokenizer_handlers: Provider[TextTokenizerHandler],
    ) -> None:
        self._asset_store = asset_store
        self._tokenizer_handlers = tokenizer_handlers

    def load(self, tokenizer_section: TextTokenizerSection) -> TextTokenizer:
        tokenizer_name = tokenizer_section.name

        try:
            card = self._asset_store.retrieve_card(tokenizer_name)
        except AssetCardNotFoundError:
            raise UnknownTextTokenizerError(tokenizer_name) from None
        except AssetCardError as ex:
            raise text_tokenizer_asset_card_error(tokenizer_name) from ex

        try:
            card = resolve_text_tokenizer_reference(self._asset_store, card)
        except AssetCardError as ex:
            raise text_tokenizer_asset_card_error(tokenizer_name) from ex

        try:
            tokenizer_family = card.field("tokenizer_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise UnknownTextTokenizerError(tokenizer_name) from None
        except AssetCardError as ex:
            raise text_tokenizer_asset_card_error(tokenizer_name) from ex

        try:
            handler = self._tokenizer_handlers.get(tokenizer_family)
        except LookupError:
            raise UnknownTextTokenizerFamilyError(
                tokenizer_family, tokenizer_name
            ) from None

        log.info("Loading '{}' tokenizer.", tokenizer_name)

        tokenizer = handler.load(card)

        log.info("Tokenizer loaded.")

        log_tokenizer(log, tokenizer)

        return tokenizer
