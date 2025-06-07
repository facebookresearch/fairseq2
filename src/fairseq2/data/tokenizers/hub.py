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
from fairseq2.data.tokenizers.error import (
    UnknownTokenizerError,
    UnknownTokenizerFamilyError,
    tokenizer_asset_card_error,
)
from fairseq2.data.tokenizers.handler import TokenizerHandler
from fairseq2.data.tokenizers.ref import resolve_tokenizer_reference
from fairseq2.data.tokenizers.tokenizer import Tokenizer
from fairseq2.dependency import DependencyResolver
from fairseq2.typing import Provider


@final
class TokenizerHub:
    _asset_store: AssetStore
    _handlers: Provider[TokenizerHandler]
    _resolver: DependencyResolver

    def __init__(
        self,
        asset_store: AssetStore,
        handlers: Provider[TokenizerHandler],
        resolver: DependencyResolver,
    ) -> None:
        self._asset_store = asset_store
        self._handlers = handlers
        self._resolver = resolver

    def load(self, name_or_card: str | AssetCard) -> Tokenizer:
        if isinstance(name_or_card, AssetCard):
            card = name_or_card

            name = card.name
        else:
            name = name_or_card

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

        return handler.load(self._resolver, card)


def tokenizer_hub() -> TokenizerHub:
    from fairseq2 import get_dependency_resolver

    resolver = get_dependency_resolver()

    asset_store = resolver.resolve(AssetStore)

    handlers = resolver.resolve_provider(TokenizerHandler)

    return TokenizerHub(asset_store, handlers, resolver)
