# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final

from typing_extensions import override

from fairseq2.assets import (
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetCardNotFoundError,
    AssetStore,
)
from fairseq2.datasets import (
    DatasetHandler,
    DatasetLoadError,
    UnknownDatasetError,
    UnknownDatasetFamilyError,
    dataset_asset_card_error,
)
from fairseq2.dependency import DependencyResolver
from fairseq2.error import SetupError
from fairseq2.gang import GangError, Gangs
from fairseq2.logging import log
from fairseq2.recipe.config import DatasetSectionBase, get_config_section
from fairseq2.recipe.error import DatasetPathNotFoundError
from fairseq2.typing import Provider


def load_dataset(resolver: DependencyResolver) -> object:
    dataset_section = get_config_section(resolver, "dataset", DatasetSectionBase)

    asset_store = resolver.resolve(AssetStore)

    handlers = resolver.resolve_provider(DatasetHandler)

    gangs = resolver.resolve(Gangs)

    loader: DatasetLoader

    if dataset_section.path is not None:
        loader = _PathBasedDatasetLoader(handlers)
    elif dataset_section.name is not None:
        loader = _CardBasedDatasetLoader(asset_store, handlers)
    else:
        raise ValueError(
            "Either `config.dataset.name` or `config.dataset.path` must be specified."
        )

    try:
        return loader.load(resolver, dataset_section, gangs)
    except DatasetLoadError as ex:
        raise SetupError(
            f"The '{ex.dataset_name}' dataset cannot be loaded. See the nested exception for details."
        ) from ex


class DatasetLoader(ABC):
    @abstractmethod
    def load(
        self,
        resolver: DependencyResolver,
        dataset_section: DatasetSectionBase,
        gangs: Gangs,
    ) -> object: ...


@final
class _CardBasedDatasetLoader(DatasetLoader):
    _asset_store: AssetStore
    _handlers: Provider[DatasetHandler]

    def __init__(
        self, asset_store: AssetStore, handlers: Provider[DatasetHandler]
    ) -> None:
        self._asset_store = asset_store
        self._handlers = handlers

    @override
    def load(
        self,
        resolver: DependencyResolver,
        dataset_section: DatasetSectionBase,
        gangs: Gangs,
    ) -> object:
        name = dataset_section.name
        if name is None:
            raise ValueError("`dataset_section.name` must be specified.")

        try:
            card = self._asset_store.retrieve_card(name)
        except AssetCardNotFoundError:
            raise UnknownDatasetError(name) from None
        except AssetCardError as ex:
            raise dataset_asset_card_error(name) from ex

        try:
            family = card.field("dataset_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise UnknownDatasetError(name) from None
        except AssetCardError as ex:
            raise dataset_asset_card_error(name) from ex

        try:
            handler = self._handlers.resolve(family)
        except LookupError:
            raise UnknownDatasetFamilyError(family, name) from None

        log.info("Loading '{}' dataset.", name)

        dataset = handler.load(resolver, card)

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise DatasetLoadError(
                name, f"The collective barrier after the '{name}' dataset load operation has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        log.info("Dataset loaded.")

        return dataset


@final
class _PathBasedDatasetLoader(DatasetLoader):
    _handlers: Provider[DatasetHandler]

    def __init__(self, handlers: Provider[DatasetHandler]) -> None:
        self._handlers = handlers

    @override
    def load(
        self,
        resolver: DependencyResolver,
        dataset_section: DatasetSectionBase,
        gangs: Gangs,
    ) -> object:
        family = dataset_section.family

        path = dataset_section.path
        if path is None:
            raise ValueError("`dataset_section.path` must be specified.")

        name = path.name

        try:
            handler = self._handlers.resolve(family)
        except LookupError:
            raise UnknownDatasetFamilyError(family, name) from None

        log.info("Loading the dataset.")

        try:
            dataset = handler.load_from_path(resolver, path, name)
        except FileNotFoundError:
            raise DatasetPathNotFoundError(name, path) from None

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise DatasetLoadError(
                name, f"The collective barrier after the '{name}' dataset load operation has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        log.info("Dataset loaded.")

        return dataset
