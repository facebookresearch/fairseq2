# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.assets import (
    AssetCardFieldNotFoundError,
    AssetNotFoundError,
    AssetStore,
)
from fairseq2.data.tokenizers import (
    Tokenizer,
    TokenizerFamilyHandler,
    TokenizerLoadError,
    UnknownTokenizerError,
    UnknownTokenizerFamilyError,
    resolve_tokenizer_reference,
)
from fairseq2.error import ContractError, InternalError
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.recipe.config import ConfigOverrider, TokenizerSection, get_config_section
from fairseq2.recipe.error import TokenizerNotFoundError
from fairseq2.recipe.utils.log import log_tokenizer
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.utils.structured import StructureError, ValueConverter
from fairseq2.utils.validation import ObjectValidator, ValidationError


def _load_tokenizer(resolver: DependencyResolver) -> Tokenizer:
    return load_tokenizer(resolver, section_name="tokenizer")


def _load_source_tokenizer(resolver: DependencyResolver) -> Tokenizer:
    return load_tokenizer(resolver, section_name="source_tokenizer")


def _load_target_tokenizer(resolver: DependencyResolver) -> Tokenizer:
    return load_tokenizer(resolver, section_name="target_tokenizer")


def load_tokenizer(resolver: DependencyResolver, section_name: str) -> Tokenizer:
    section = get_config_section(resolver, section_name, TokenizerSection)

    if section.path is not None:
        return _load_tokenizer_from_path(resolver, section_name, section)

    if section.name is not None:
        return _load_tokenizer_from_card(resolver, section_name, section)

    raise InternalError("`section.name` or `section.path` are both `None`.")


def _load_tokenizer_from_card(
    resolver: DependencyResolver, section_name: str, section: TokenizerSection
) -> Tokenizer:
    value_converter = resolver.resolve(ValueConverter)

    asset_store = resolver.resolve(AssetStore)

    handlers = resolver.get_provider(TokenizerFamilyHandler)

    gangs = resolver.resolve(Gangs)

    validator = resolver.resolve(ObjectValidator)

    config_overrider = ConfigOverrider(value_converter)

    name = section.name
    if name is None:
        raise InternalError("`tokenizer_section.name` is `None`.")

    try:
        card = asset_store.retrieve_card(name)
    except AssetNotFoundError:
        raise UnknownTokenizerError(name) from None

    card = resolve_tokenizer_reference(asset_store, card)

    try:
        family = card.field("tokenizer_family").as_(str)
    except AssetCardFieldNotFoundError:
        raise UnknownTokenizerError(name) from None

    try:
        try:
            handler = handlers.get(family)
        except LookupError:
            raise UnknownTokenizerFamilyError(family) from None
    except UnknownTokenizerFamilyError as ex:
        raise TokenizerLoadError(
            name, f"The '{family}' family of the '{name}' tokenizer is not known."  # fmt: skip
        ) from ex

    config = handler.load_tokenizer_config(card)

    try:
        config = config_overrider.apply_overrides(config, section.config_overrides)
    except StructureError as ex:
        raise StructureError(
            f"`{section_name}.config_overrides` cannot be structured. See the nested exception for details."
        ) from ex

    try:
        validator.validate(config)
    except ValidationError as ex:
        raise ValidationError(
            ex.result, field=f"{section_name}.config_overrides"
        ) from None

    log.info("Loading '{}' tokenizer.", name)

    try:
        tokenizer = handler.load_tokenizer(resolver, card, config=config)
    except ValueError as ex:
        raise TokenizerLoadError(
            name, f"The '{name}' tokenizer does not have a valid configuration. See the nested exception for details."  # fmt: skip
        ) from ex

    gangs.root.barrier()

    log.info("Tokenizer loaded.")

    log_tokenizer(tokenizer)

    return tokenizer


def _load_tokenizer_from_path(
    resolver: DependencyResolver, section_name: str, section: TokenizerSection
) -> Tokenizer:
    value_converter = resolver.resolve(ValueConverter)

    handlers = resolver.get_provider(TokenizerFamilyHandler)

    gangs = resolver.resolve(Gangs)

    validator = resolver.resolve(ObjectValidator)

    config_overrider = ConfigOverrider(value_converter)

    path = section.path
    if path is None:
        raise InternalError("`section.path` is `None`.")

    family = section.family
    if family is None:
        raise InternalError("`section.family` is `None`.")

    name = str(path)

    try:
        handler = handlers.get(family)
    except LookupError:
        raise UnknownTokenizerFamilyError(family) from None

    try:
        config = handler.config_kls()
    except TypeError as ex:
        raise ContractError(
            f"The default configuration of the '{family}' tokenizer family cannot be constructed. See the nested exception for details."
        ) from ex

    try:
        config = config_overrider.apply_overrides(config, section.config_overrides)
    except StructureError as ex:
        raise StructureError(
            f"`{section_name}.config_overrides` cannot be structured. See the nested exception for details."
        ) from ex

    try:
        validator.validate(config)
    except ValidationError as ex:
        raise ValidationError(
            ex.result, field=f"{section_name}.config_overrides"
        ) from None

    log.info("Loading '{}' tokenizer.", name)

    try:
        tokenizer = handler.load_tokenizer_from_path(resolver, path, config)
    except ValueError as ex:
        raise TokenizerLoadError(
            name, f"The '{name}' tokenizer does not have a valid configuration. See the nested exception for details."  # fmt: skip
        ) from ex
    except FileNotFoundError as ex:
        raise TokenizerNotFoundError(path) from ex

    gangs.root.barrier()

    log.info("Tokenizer loaded.")

    log_tokenizer(tokenizer)

    return tokenizer
