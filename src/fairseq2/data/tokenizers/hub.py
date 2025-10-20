# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Generic, TypeVar, cast, final

from fairseq2.assets import AssetCard, AssetCardError, AssetNotFoundError, AssetStore
from fairseq2.data.tokenizers.family import TokenizerFamily
from fairseq2.data.tokenizers.ref import resolve_tokenizer_reference
from fairseq2.data.tokenizers.tokenizer import Tokenizer
from fairseq2.error import InternalError
from fairseq2.runtime.dependency import get_dependency_resolver
from fairseq2.runtime.lookup import Lookup
from fairseq2.utils.warn import _warn_progress_deprecated

TokenizerT = TypeVar("TokenizerT", bound=Tokenizer)

TokenizerConfigT = TypeVar("TokenizerConfigT")


@final
class TokenizerHub(Generic[TokenizerT, TokenizerConfigT]):
    def __init__(self, family: TokenizerFamily, asset_store: AssetStore) -> None:
        self._family = family
        self._asset_store = asset_store

    def iter_cards(self) -> Iterator[AssetCard]:
        return self._asset_store.find_cards("tokenizer_family", self._family.name)

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

        family_name = card.field("tokenizer_family").as_(str)

        if family_name != self._family.name:
            msg = f"family field of the {name} asset card is expected to be {self._family.name}, but is {family_name} instead."

            raise AssetCardError(name, msg)

        config = self._family.get_tokenizer_config(card)

        return cast(TokenizerConfigT, config)

    def load_tokenizer(
        self,
        card: AssetCard | str,
        *,
        config: TokenizerConfigT | None = None,
        progress: bool | None = None,
    ) -> TokenizerT:
        _warn_progress_deprecated(progress)

        if isinstance(card, str):
            name = card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError:
                raise TokenizerNotKnownError(name) from None
        else:
            name = card.name

        card = resolve_tokenizer_reference(self._asset_store, card)

        family_name = card.field("tokenizer_family").as_(str)

        if family_name != self._family.name:
            msg = f"family field of the {name} asset card is expected to be {self._family.name}, but is {family_name} instead."

            raise AssetCardError(name, msg)

        tokenizer = self._family.load_tokenizer(card, config, progress=True)

        return cast(TokenizerT, tokenizer)

    def load_custom_tokenizer(self, path: Path, config: TokenizerConfigT) -> TokenizerT:
        tokenizer = self._family.load_custom_tokenizer(path, config)

        return cast(TokenizerT, tokenizer)


@final
class TokenizerHubAccessor(Generic[TokenizerT, TokenizerConfigT]):
    def __init__(
        self,
        family_name: str,
        kls: type[TokenizerT],
        config_kls: type[TokenizerConfigT],
    ) -> None:
        self._family_name = family_name
        self._kls = kls
        self._config_kls = config_kls

    def __call__(self) -> TokenizerHub[TokenizerT, TokenizerConfigT]:
        resolver = get_dependency_resolver()

        asset_store = resolver.resolve(AssetStore)

        name = self._family_name

        family = resolver.resolve_optional(TokenizerFamily, key=name)
        if family is None:
            raise TokenizerFamilyNotKnownError(name)

        if not issubclass(family.kls, self._kls):
            raise InternalError(
                f"`kls` is `{self._kls}`, but the type of the {name} tokenizer family is `{family.kls}`."
            )

        if not issubclass(family.config_kls, self._config_kls):
            raise InternalError(
                f"`config_kls` is `{self._config_kls}`, but the configuration type of the {name} tokenizer family is `{family.config_kls}`."
            )

        return TokenizerHub(family, asset_store)


class TokenizerNotKnownError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is not a known tokenizer.")

        self.name = name


class TokenizerFamilyNotKnownError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is not a known tokenizer family.")

        self.name = name


def load_tokenizer(
    card: AssetCard | str, *, config: object | None = None, progress: bool | None = None
) -> Tokenizer:
    resolver = get_dependency_resolver()

    global_loader = resolver.resolve(GlobalTokenizerLoader)

    return global_loader.load(card, config, progress)


@final
class GlobalTokenizerLoader:
    def __init__(
        self, asset_store: AssetStore, families: Lookup[TokenizerFamily]
    ) -> None:
        self._asset_store = asset_store
        self._families = families

    def load(
        self, card: AssetCard | str, config: object | None, progress: bool | None
    ) -> Tokenizer:
        _warn_progress_deprecated(progress)

        if isinstance(card, str):
            name = card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError:
                raise TokenizerNotKnownError(name) from None
        else:
            name = card.name

        card = resolve_tokenizer_reference(self._asset_store, card)

        family_name = card.field("tokenizer_family").as_(str)

        family = self._families.maybe_get(family_name)
        if family is None:
            msg = f"family field of the {name} asset card is expected to be a supported tokenizer family, but is {family_name} instead."

            raise AssetCardError(name, msg)

        return family.load_tokenizer(card, config, progress=True)
