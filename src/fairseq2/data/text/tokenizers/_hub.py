# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from fairseq2.assets import AssetCard, AssetStore
from fairseq2.context import get_runtime_context
from fairseq2.data.text.tokenizers._handler import (
    TextTokenizerHandler,
    TextTokenizerNotFoundError,
    get_text_tokenizer_family,
)
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
        else:
            card = self._asset_store.retrieve_card(name_or_card)

        card = resolve_text_tokenizer_reference(self._asset_store, card)

        family = get_text_tokenizer_family(card)

        try:
            handler = self._tokenizer_handlers.get(family)
        except LookupError:
            raise TextTokenizerNotFoundError(card.name) from None

        return handler.load(card)


def get_text_tokenizer_hub() -> TextTokenizerHub:
    context = get_runtime_context()

    tokenizer_handlers = context.get_registry(TextTokenizerHandler)

    return TextTokenizerHub(context.asset_store, tokenizer_handlers)
