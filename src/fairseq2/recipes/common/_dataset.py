# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar, cast, final

from typing_extensions import override

from fairseq2.assets import (
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetCardNotFoundError,
    AssetStore,
)
from fairseq2.context import RuntimeContext
from fairseq2.datasets import (
    DatasetHandler,
    DatasetLoadError,
    InvalidDatasetTypeError,
    UnknownDatasetError,
    UnknownDatasetFamilyError,
    dataset_asset_card_error,
)
from fairseq2.gang import GangError, Gangs
from fairseq2.logging import log
from fairseq2.recipes import RecipeError
from fairseq2.recipes.common._error import DatasetPathNotFoundError
from fairseq2.recipes.config import DatasetSection, get_config_section
from fairseq2.registry import Provider

DatasetT = TypeVar("DatasetT")


def load_dataset(
    kls: type[DatasetT], context: RuntimeContext, recipe_config: object, gangs: Gangs
) -> DatasetT:
    dataset_handlers = context.get_registry(DatasetHandler)

    dataset_loader: DatasetLoader

    dataset_section = get_config_section(recipe_config, "dataset", DatasetSection)
    if dataset_section.path is not None:
        dataset_loader = PathBasedDatasetLoader(kls, dataset_handlers)
    elif dataset_section.name is not None:
        dataset_loader = CardBasedDatasetLoader(
            kls, context.asset_store, dataset_handlers
        )
    else:
        raise ValueError(
            "Either `config.dataset.name` or `config.dataset.path` must be specified."
        )

    try:
        dataset = dataset_loader.load(recipe_config, gangs)
    except DatasetLoadError as ex:
        raise RecipeError(
            f"The '{ex.dataset_name}' dataset cannot be loaded. See the nested exception for details."
        ) from ex

    return cast(DatasetT, dataset)


class DatasetLoader(ABC):
    @abstractmethod
    def load(self, recipe_config: object, gangs: Gangs) -> object: ...


@final
class CardBasedDatasetLoader(DatasetLoader):
    _kls: type[object]
    _asset_store: AssetStore
    _dataset_handlers: Provider[DatasetHandler]

    def __init__(
        self,
        kls: type[object],
        asset_store: AssetStore,
        dataset_handlers: Provider[DatasetHandler],
    ) -> None:
        self._kls = kls
        self._asset_store = asset_store
        self._dataset_handlers = dataset_handlers

    @override
    def load(self, recipe_config: object, gangs: Gangs) -> object:
        dataset_section = get_config_section(recipe_config, "dataset", DatasetSection)

        dataset_name = dataset_section.name
        if dataset_name is None:
            raise ValueError("`recipe_config.dataset.name` must be specified.")

        try:
            card = self._asset_store.retrieve_card(dataset_name)
        except AssetCardNotFoundError:
            raise UnknownDatasetError(dataset_name) from None
        except AssetCardError as ex:
            raise dataset_asset_card_error(dataset_name) from ex

        try:
            dataset_family = card.field("dataset_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise UnknownDatasetError(dataset_name) from None
        except AssetCardError as ex:
            raise dataset_asset_card_error(dataset_name) from ex

        try:
            handler = self._dataset_handlers.get(dataset_family)
        except LookupError:
            raise UnknownDatasetFamilyError(dataset_family, dataset_name) from None

        if not issubclass(handler.kls, self._kls):
            raise InvalidDatasetTypeError(dataset_name, handler.kls, self._kls)

        log.info("Loading '{}' dataset.", dataset_name)

        dataset = handler.load(card)

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise DatasetLoadError(
                dataset_name, f"The collective barrier after the '{dataset_name}' dataset load operation has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        log.info("Dataset loaded.")

        return dataset


@final
class PathBasedDatasetLoader(DatasetLoader):
    _kls: type[object]
    _dataset_handlers: Provider[DatasetHandler]

    def __init__(
        self, kls: type[object], dataset_handlers: Provider[DatasetHandler]
    ) -> None:
        self._kls = kls
        self._dataset_handlers = dataset_handlers

    @override
    def load(self, recipe_config: object, gangs: Gangs) -> object:
        dataset_section = get_config_section(recipe_config, "dataset", DatasetSection)

        dataset_family = dataset_section.family

        data_path = dataset_section.path
        if data_path is None:
            raise ValueError("`recipe.dataset.path` must be specified.")

        dataset_name = "custom"

        try:
            handler = self._dataset_handlers.get(dataset_family)
        except LookupError:
            raise UnknownDatasetFamilyError(dataset_family, dataset_name) from None

        if not issubclass(handler.kls, self._kls):
            raise InvalidDatasetTypeError(dataset_name, handler.kls, self._kls)

        log.info("Loading the dataset.")

        try:
            dataset = handler.load_from_path(data_path, dataset_name)
        except FileNotFoundError:
            raise DatasetPathNotFoundError(dataset_name, data_path) from None

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise DatasetLoadError(
                dataset_name, f"The collective barrier after the '{dataset_name}' dataset load operation has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        log.info("Dataset loaded.")

        return dataset
