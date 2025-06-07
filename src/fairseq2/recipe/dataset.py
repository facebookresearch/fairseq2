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
from fairseq2.datasets import (
    DatasetError,
    DatasetFamilyHandler,
    UnknownDatasetError,
    UnknownDatasetFamilyError,
)
from fairseq2.error import InternalError
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.recipe.config import DatasetSectionBase, get_config_section
from fairseq2.recipe.error import DatasetNotFoundError
from fairseq2.runtime.dependency import DependencyResolver


def _open_dataset(resolver: DependencyResolver) -> object:
    section = get_config_section(resolver, "dataset", DatasetSectionBase)

    if section.path is not None:
        return _open_dataset_from_path(resolver, section)

    if section.name is not None:
        return _open_dataset_from_card(resolver, section)

    raise InternalError("`section.name` and `section.path` are both `None`.")


def _open_dataset_from_card(
    resolver: DependencyResolver, section: DatasetSectionBase
) -> object:
    asset_store = resolver.resolve(AssetStore)

    handlers = resolver.get_provider(DatasetFamilyHandler)

    gangs = resolver.resolve(Gangs)

    name = section.name
    if name is None:
        raise InternalError("`section.name` is `None`.")

    try:
        card = asset_store.retrieve_card(name)
    except AssetNotFoundError:
        raise UnknownDatasetError(name) from None

    try:
        family = card.field("dataset_family").as_(str)
    except AssetCardFieldNotFoundError:
        raise UnknownDatasetError(name) from None

    try:
        try:
            handler = handlers.get(family)
        except LookupError:
            raise UnknownDatasetFamilyError(family) from None
    except UnknownDatasetFamilyError as ex:
        raise DatasetError(
            name, f"The '{family}' family of the '{name}' tokenizer is not known."  # fmt: skip
        ) from ex

    log.info("Loading '{}' dataset.", name)

    try:
        dataset = handler.open_dataset(resolver, card)
    except ValueError as ex:
        raise DatasetError(
            name, f"The '{name}' dataset does not have a valid configuration. See the nested exception for details."  # fmt: skip
        ) from ex

    gangs.root.barrier()

    log.info("Dataset loaded.")

    return dataset


def _open_dataset_from_path(
    resolver: DependencyResolver, section: DatasetSectionBase
) -> object:
    handlers = resolver.get_provider(DatasetFamilyHandler)

    gangs = resolver.resolve(Gangs)

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
        raise UnknownDatasetFamilyError(family) from None

    log.info("Loading '{}' dataset.", name)

    try:
        dataset = handler.open_dataset_from_path(resolver, path)
    except ValueError as ex:
        raise DatasetError(
            name, f"The '{name}' dataset does not have a valid configuration. See the nested exception for details."  # fmt: skip
        ) from ex
    except FileNotFoundError as ex:
        raise DatasetNotFoundError(path) from ex

    gangs.root.barrier()

    log.info("Dataset loaded.")

    return dataset
