# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetCardNotFoundError,
    AssetStore,
)
from fairseq2.context import get_runtime_context
from fairseq2.data.text.tokenizers._error import (
    UnknownTextTokenizerError,
    UnknownTextTokenizerFamilyError,
    text_tokenizer_asset_card_error,
)
from fairseq2.data.text.tokenizers._handler import TextTokenizerHandler
from fairseq2.data.text.tokenizers._ref import resolve_text_tokenizer_reference
from fairseq2.data.text.tokenizers._tokenizer import TextTokenizer
from fairseq2.registry import Provider


@final
class TextTokenizerHub:
    _asset_store: AssetStore
    _tokenizer_handlers: Provider[TextTokenizerHandler]

    def __init__(
        self,
        asset_store: AssetStore,
        tokenizer_handlers: Provider[TextTokenizerHandler],
    ) -> None:
        self._asset_store = asset_store
        self._tokenizer_handlers = tokenizer_handlers

    def load(self, name_or_card: str | AssetCard) -> TextTokenizer:
        if isinstance(name_or_card, AssetCard):
            card = name_or_card

            tokenizer_name = card.name
        else:
            tokenizer_name = name_or_card

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

        return handler.load(card)


def get_text_tokenizer_hub() -> TextTokenizerHub:
    context = get_runtime_context()

    tokenizer_handlers = context.get_registry(TextTokenizerHandler)

    return TextTokenizerHub(context.asset_store, tokenizer_handlers)
