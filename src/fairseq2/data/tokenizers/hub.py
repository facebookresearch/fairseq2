# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Generic, TypeVar, cast, final

from fairseq2.assets import AssetCard, AssetCardError, AssetNotFoundError, AssetStore
from fairseq2.data.tokenizers.handler import TokenizerFamilyHandler
from fairseq2.data.tokenizers.ref import resolve_tokenizer_reference
from fairseq2.data.tokenizers.tokenizer import Tokenizer
from fairseq2.error import InternalError
from fairseq2.runtime.dependency import get_dependency_resolver
from fairseq2.runtime.provider import Provider

TokenizerConfigT = TypeVar("TokenizerConfigT")


@final
class TokenizerHub(Generic[TokenizerConfigT]):
    def __init__(
        self, handler: TokenizerFamilyHandler, asset_store: AssetStore
    ) -> None:
        self._handler = handler
        self._asset_store = asset_store

    def iter_cards(self) -> Iterable[AssetCard]:
        return self._asset_store.find_cards("tokenizer_family", self._handler.family)

    def get_tokenizer_config(self, card: AssetCard | str) -> TokenizerConfigT:
        if isinstance(card, str):
            name = card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError:
                raise TokenizerNotKnownError(name) from None
        else:
            name = card.name

        card = resolve_tokenizer_reference(self._asset_store, card)

        family = card.field("tokenizer_family").as_(str)

        if family != self._handler.family:
            msg = f"family field of the {name} asset card is expected to be '{self._handler.family}', but is '{family}' instead."

            raise AssetCardError(name, msg)

        config = self._handler.get_tokenizer_config(card)

        return cast(TokenizerConfigT, config)

    def load_tokenizer(
        self,
        card: AssetCard | str,
        *,
        config: TokenizerConfigT | None = None,
        progress: bool = False,
    ) -> Tokenizer:
        if isinstance(card, str):
            name = card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError:
                raise TokenizerNotKnownError(name) from None
        else:
            name = card.name

        card = resolve_tokenizer_reference(self._asset_store, card)

        family = card.field("tokenizer_family").as_(str)

        if family != self._handler.family:
            msg = f"family field of the {name} asset card is expected to be {self._handler.family}, but is {family} instead."

            raise AssetCardError(name, msg)

        return self._handler.load_tokenizer(card, config, progress)

    def load_custom_tokenizer(self, path: Path, config: TokenizerConfigT) -> Tokenizer:
        return self._handler.load_custom_tokenizer(path, config)

    @property
    def handler(self) -> TokenizerFamilyHandler:
        return self._handler


@final
class TokenizerHubAccessor(Generic[TokenizerConfigT]):
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
        except LookupError:
            raise TokenizerFamilyNotKnownError(family) from None

        if not issubclass(handler.config_kls, self._config_kls):
            raise InternalError(
                f"`config_kls` is `{self._config_kls}`, but the configuration type of the {family} tokenizer family is `{handler.config_kls}`."
            )

        return TokenizerHub(handler, asset_store)


class TokenizerNotKnownError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is not a known tokenizer.")

        self.name = name


class TokenizerFamilyNotKnownError(Exception):
    def __init__(self, family: str) -> None:
        super().__init__(f"{family} is not a known tokenizer family.")

        self.family = family


def load_tokenizer(
    card: AssetCard | str, *, config: object = None, progress: bool = False
) -> Tokenizer:
    resolver = get_dependency_resolver()

    asset_store = resolver.resolve(AssetStore)

    handlers = resolver.get_provider(TokenizerFamilyHandler)

    loader = GlobalTokenizerLoader(asset_store, handlers)

    return loader.load(card, config, progress)


@final
class GlobalTokenizerLoader:
    def __init__(
        self, asset_store: AssetStore, handlers: Provider[TokenizerFamilyHandler]
    ) -> None:
        self._asset_store = asset_store
        self._handlers = handlers

    def load(
        self, card: AssetCard | str, config: object | None, progress: bool
    ) -> Tokenizer:
        if isinstance(card, str):
            name = card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError:
                raise TokenizerNotKnownError(name) from None
        else:
            name = card.name

        card = resolve_tokenizer_reference(self._asset_store, card)

        family = card.field("tokenizer_family").as_(str)

        handler = self._handlers.maybe_get(family)
        if handler is None:
            msg = f"family field of the {name} asset card is expected to be a supported tokenizer family, but is '{family}' instead."

            raise AssetCardError(name, msg)

        return handler.load_tokenizer(card, config, progress)
