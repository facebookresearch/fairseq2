# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import final

from fairseq2.assets import AssetCardError, AssetStore
from fairseq2.data.tokenizers import (
    Tokenizer,
    TokenizerFamilyHandler,
    TokenizerFamilyNotKnownError,
    TokenizerNotKnownError,
    resolve_tokenizer_reference,
)
from fairseq2.error import InternalError, raise_operational_system_error
from fairseq2.gang import GangError, Gangs, raise_operational_gang_error
from fairseq2.logging import log
from fairseq2.recipe.asset_config import AssetConfigOverrider
from fairseq2.recipe.config import TokenizerSection, get_config_section
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.runtime.provider import Provider
from fairseq2.utils.log import log_tokenizer


def load_tokenizer(resolver: DependencyResolver, section_name: str) -> Tokenizer:
    section = get_config_section(resolver, section_name, TokenizerSection)

    factory = resolver.resolve(TokenizerFactory)

    return factory.create(section_name, section)


@final
class TokenizerFactory:
    def __init__(
        self,
        handlers: Provider[TokenizerFamilyHandler],
        asset_store: AssetStore,
        asset_config_overrider: AssetConfigOverrider,
        gangs: Gangs,
    ) -> None:
        self._handlers = handlers
        self._asset_store = asset_store
        self._asset_config_overrider = asset_config_overrider
        self._gangs = gangs

    def create(self, section_name: str, section: TokenizerSection) -> Tokenizer:
        if section.path is not None:
            return self._load_tokenizer_from_path(section_name, section)

        if section.name is not None:
            return self._load_tokenizer_from_card(section_name, section)

        raise InternalError("`section.name` and `section.path` are both `None`.")

    def _load_tokenizer_from_card(
        self, section_name: str, section: TokenizerSection
    ) -> Tokenizer:
        name = section.name
        if name is None:
            raise InternalError("`section.name` is `None`.")

        card = self._asset_store.maybe_retrieve_card(name)
        if card is None:
            raise TokenizerNotKnownError(name)

        card = resolve_tokenizer_reference(self._asset_store, card)

        family = card.field("tokenizer_family").as_(str)

        handler = self._handlers.maybe_get(family)
        if handler is None:
            msg = f"family field of the {name} asset card is expected to be a supported tokenizer family, but is {family} instead."

            raise AssetCardError(name, msg)

        config = handler.get_tokenizer_config(card)

        if section.config_overrides is not None:
            config = self._asset_config_overrider.apply_overrides(
                section_name, config, section.config_overrides
            )

        log.info("Loading {} tokenizer.", name)

        tokenizer = handler.load_tokenizer(card, config, progress=True)

        try:
            self._gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

        log.info("Tokenizer loaded.")

        log_tokenizer(tokenizer)

        return tokenizer

    def _load_tokenizer_from_path(
        self, section_name: str, section: TokenizerSection
    ) -> Tokenizer:
        path = section.path
        if path is None:
            raise InternalError("`section.path` is `None`.")

        family = section.family
        if family is None:
            raise InternalError("`section.family` is `None`.")

        name = path.name

        handler = self._handlers.maybe_get(family)
        if handler is None:
            raise TokenizerFamilyNotKnownError(family)

        try:
            config = handler.config_kls()
        except TypeError as ex:
            raise InternalError(
                f"Default configuration of the {family} tokenizer family cannot be constructed."
            ) from ex

        if section.config_overrides is not None:
            config = self._asset_config_overrider.apply_overrides(
                section_name, config, section.config_overrides
            )

        log.info("Loading {} tokenizer.", name)

        try:
            tokenizer = handler.load_custom_tokenizer(path, config)
        except FileNotFoundError as ex:
            raise TokenizerModelNotFoundError(path) from ex
        except OSError as ex:
            raise_operational_system_error(ex)

        try:
            self._gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

        log.info("Tokenizer loaded.")

        log_tokenizer(tokenizer)

        return tokenizer


class TokenizerModelNotFoundError(Exception):
    def __init__(self, path: Path) -> None:
        super().__init__(f"{path} does not point to a tokenizer model.")

        self.path = path
