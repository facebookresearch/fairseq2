# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Generic, TypeVar, cast, final

from fairseq2.assets import (
    AssetCard,
    AssetCardFieldNotFoundError,
    AssetNotFoundError,
    AssetStore,
)
from fairseq2.data.tokenizers.error import (
    TokenizerConfigLoadError,
    TokenizerLoadError,
    UnknownTokenizerError,
    UnknownTokenizerFamilyError,
)
from fairseq2.data.tokenizers.handler import TokenizerFamilyHandler
from fairseq2.data.tokenizers.ref import resolve_tokenizer_reference
from fairseq2.data.tokenizers.tokenizer import Tokenizer
from fairseq2.error import ContractError
from fairseq2.runtime.dependency import (
    DependencyNotFoundError,
    DependencyResolver,
    get_dependency_resolver,
)

TokenizerConfigT = TypeVar("TokenizerConfigT")


@final
class TokenizerHub(Generic[TokenizerConfigT]):
    _handler: TokenizerFamilyHandler
    _asset_store: AssetStore
    _resolver: DependencyResolver

    def __init__(
        self,
        handler: TokenizerFamilyHandler,
        asset_store: AssetStore,
        resolver: DependencyResolver,
    ) -> None:
        self._handler = handler
        self._asset_store = asset_store
        self._resolver = resolver

    def iter_tokenizer_cards(self) -> Iterable[AssetCard]:
        return self._asset_store.find_cards("tokenizer_family", self._handler.family)

    def load_tokenizer_config(self, name_or_card: str | AssetCard) -> TokenizerConfigT:
        if isinstance(name_or_card, AssetCard):
            card = name_or_card

            name = card.name
        else:
            name = name_or_card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError:
                raise UnknownTokenizerError(name) from None

        card = resolve_tokenizer_reference(self._asset_store, card)

        try:
            family = card.field("tokenizer_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise UnknownTokenizerError(name) from None

        if family != self._handler.family:
            raise TokenizerConfigLoadError(
                name, f"The '{name}' tokenizer does not belong to the '{family}' family."  # fmt: skip
            )

        config = self._handler.load_tokenizer_config(card)

        return cast(TokenizerConfigT, config)

    def load_tokenizer(
        self, name_or_card: str | AssetCard, *, config: TokenizerConfigT | None = None
    ) -> Tokenizer:
        if isinstance(name_or_card, AssetCard):
            card = name_or_card

            name = card.name
        else:
            name = name_or_card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError:
                raise UnknownTokenizerError(name) from None

        card = resolve_tokenizer_reference(self._asset_store, card)

        try:
            family = card.field("tokenizer_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise UnknownTokenizerError(name) from None

        if family != self._handler.family:
            raise TokenizerLoadError(
                name, f"The '{name}' tokenizer does not belong to the '{family}' family."  # fmt: skip
            )

        return self._handler.load_tokenizer(self._resolver, card, config=config)

    def load_tokenizer_from_path(
        self, path: Path, config: TokenizerConfigT
    ) -> Tokenizer:
        return self._handler.load_tokenizer_from_path(self._resolver, path, config)

    @property
    def handler(self) -> TokenizerFamilyHandler:
        return self._handler


@final
class TokenizerHubAccessor(Generic[TokenizerConfigT]):
    _family: str
    _config_kls: type[TokenizerConfigT]

    def __init__(self, family: str, config_kls: type[TokenizerConfigT]) -> None:
        self._family = family
        self._config_kls = config_kls

    def __call__(self) -> TokenizerHub[TokenizerConfigT]:
        resolver = get_dependency_resolver()

        asset_store = resolver.resolve(AssetStore)

        handlers = resolver.get_provider(TokenizerFamilyHandler)

        family = self._family

        try:
            handler = handlers.get(family)
        except DependencyNotFoundError:
            raise UnknownTokenizerFamilyError(family) from None

        if not issubclass(handler.config_kls, self._config_kls):
            raise ContractError(
                f"`config_kls` is `{self._config_kls}`, but the configuration type of the '{family}' tokenizer family is `{handler.config_kls}`."
            )

        return TokenizerHub(handler, asset_store, resolver)


def get_tokenizer_hub(family: str) -> TokenizerHub[object]:
    accessor = TokenizerHubAccessor(family, object)

    return accessor()
