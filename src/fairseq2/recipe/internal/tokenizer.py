# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from fairseq2.assets import AssetStore
from fairseq2.data.tokenizers import (
    Tokenizer,
    TokenizerFamily,
    TokenizerFamilyNotKnownError,
    TokenizerNotKnownError,
    get_tokenizer_family,
    resolve_tokenizer_reference,
)
from fairseq2.error import InternalError, raise_operational_system_error
from fairseq2.gang import GangError, Gangs, raise_operational_gang_error
from fairseq2.logging import log
from fairseq2.recipe.config import TokenizerSection
from fairseq2.recipe.error import ErrorContext, TokenizerModelNotFoundError
from fairseq2.recipe.internal.asset_config import _AssetConfigOverrider
from fairseq2.recipe.internal.log import _LogHelper
from fairseq2.recipe.tokenizer import RecipeTokenizer
from fairseq2.runtime.lookup import Lookup


@final
class _RecipeTokenizerLoader:
    def __init__(
        self,
        families: Lookup[TokenizerFamily],
        asset_store: AssetStore,
        asset_config_overrider: _AssetConfigOverrider,
        gangs: Gangs,
        log_helper: _LogHelper,
    ) -> None:
        self._families = families
        self._asset_store = asset_store
        self._asset_config_overrider = asset_config_overrider
        self._gangs = gangs
        self._log_helper = log_helper

    def load(self, section_name: str, section: TokenizerSection) -> RecipeTokenizer:
        try:
            if section.path is not None:
                if section.name is not None:
                    log.warning("Both `{0}.name` and `{0}.path` are specified. `{0}.path` takes precedence.", section_name)  # fmt: skip

                return self._load_custom_tokenizer(section_name, section)

            if section.name is not None:
                if section.family is not None:
                    log.warning("`{0}.family` will be ignored since `{0}.name` is specified.", section_name)  # fmt: skip

                return self._load_tokenizer(section_name, section)
        except Exception as ex:
            ErrorContext.set_config_section_name(ex, section_name)

            raise

        raise InternalError("`section.name` and `section.path` are both `None`.")

    def _load_tokenizer(
        self, section_name: str, section: TokenizerSection
    ) -> RecipeTokenizer:
        name = section.name
        if name is None:
            raise InternalError("`section.name` is `None`.")

        card = self._asset_store.maybe_retrieve_card(name)
        if card is None:
            raise TokenizerNotKnownError(name)

        card = resolve_tokenizer_reference(self._asset_store, card)

        family = get_tokenizer_family(card, self._families)

        config = family.get_tokenizer_config(card)

        if section.config_overrides is not None:
            config = self._asset_config_overrider.apply_overrides(
                section_name, config, section.config_overrides
            )

        if section_name == "tokenizer":
            log.info("Loading {} tokenizer.", name)
        else:
            log.info("Loading {} tokenizer specified in `{}` section.", name, section_name)  # fmt: skip

        if config is not None:
            self._log_helper.log_config("Tokenizer Config", config)

        inner_tokenizer = family.load_tokenizer(card, config, progress=True)

        try:
            self._gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

        log.info("Tokenizer loaded.")

        _log_tokenizer(inner_tokenizer)

        return RecipeTokenizer(inner_tokenizer, config, family)

    def _load_custom_tokenizer(
        self, section_name: str, section: TokenizerSection
    ) -> RecipeTokenizer:
        path = section.path
        if path is None:
            raise InternalError("`section.path` is `None`.")

        family_name = section.family
        if family_name is None:
            raise InternalError("`section.family` is `None`.")

        family = self._families.maybe_get(family_name)
        if family is None:
            raise TokenizerFamilyNotKnownError(family_name)

        try:
            config = family.config_kls()
        except TypeError as ex:
            raise InternalError(
                f"Default configuration of the {family_name} tokenizer family cannot be constructed."
            ) from ex

        if section.config_overrides is not None:
            config = self._asset_config_overrider.apply_overrides(
                section_name, config, section.config_overrides
            )

        if section_name == "tokenizer":
            log.info("Loading tokenizer.")
        else:
            log.info("Loading tokenizer specified in `{}` section.", section_name)

        if config is not None:
            self._log_helper.log_config("Tokenizer Config", config)

        try:
            inner_tokenizer = family.load_custom_tokenizer(path, config)
        except FileNotFoundError as ex:
            raise TokenizerModelNotFoundError(path) from ex
        except OSError as ex:
            raise_operational_system_error(ex)

        try:
            self._gangs.root.barrier()
        except GangError as ex:
            raise_operational_gang_error(ex)

        log.info("Tokenizer loaded.")

        _log_tokenizer(inner_tokenizer)

        return RecipeTokenizer(inner_tokenizer, config, family)


def _log_tokenizer(tokenizer: Tokenizer) -> None:
    if not log.is_enabled_for_info():
        return

    vi = tokenizer.vocab_info

    s = (
        f"Size: {vi.size:,} | "
        f"UNK: {vi.unk_idx} | "
        f"BOS: {vi.bos_idx} | "
        f"EOS: {vi.eos_idx} | "
        f"PAD: {vi.pad_idx} | "
        f"BOH: {vi.boh_idx} | "
        f"EOH: {vi.eoh_idx}"
    )

    log.info("Tokenizer - {}", s)
