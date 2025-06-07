# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, TypeVar, final

from typing_extensions import override

from fairseq2.assets import AssetCard, AssetCardError, AssetConfigLoader
from fairseq2.error import InternalError, raise_operational_system_error
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


class DatasetFamilyHandler(ABC):
    @abstractmethod
    def get_dataset_config(self, card: AssetCard) -> object: ...

    @abstractmethod
    def open_dataset(self, card: AssetCard, config: object | None) -> object: ...

    @abstractmethod
    def open_custom_dataset(self, name: str, config: object) -> object: ...

    @property
    @abstractmethod
    def family(self) -> str: ...

    @property
    @abstractmethod
    def kls(self) -> type[object]: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...


class DatasetOpenError(Exception):
    def __init__(self, dataset_name: str, message: str) -> None:
        super().__init__(message)

        self.dataset_name = dataset_name


DatasetConfigT_contra = TypeVar("DatasetConfigT_contra", contravariant=True)


class DatasetOpener(Protocol[DatasetConfigT_contra]):
    def __call__(self, name: str, config: DatasetConfigT_contra) -> object: ...


DatasetConfigT = TypeVar("DatasetConfigT")


@final
class StandardDatasetFamilyHandler(DatasetFamilyHandler):
    _config_kls: type[object]
    _opener: DatasetOpener[Any]

    def __init__(
        self,
        family: str,
        kls: type[object],
        config_kls: type[DatasetConfigT],
        opener: DatasetOpener[DatasetConfigT],
        asset_config_loader: AssetConfigLoader,
    ) -> None:
        self._family = family
        self._kls = kls
        self._config_kls = config_kls
        self._opener = opener
        self._asset_config_loader = asset_config_loader

    @override
    def get_dataset_config(self, card: AssetCard) -> object:
        try:
            default_config = self._config_kls()
        except TypeError as ex:
            raise InternalError(
                f"Default configuration of the {self._family} dataset family cannot be constructed."
            ) from ex

        return self._asset_config_loader.load(
            card, default_config, config_key="dataset_config"
        )

    @override
    def open_dataset(self, card: AssetCard, config: object | None) -> object:
        name = card.name

        # Load the configuration.
        if config is None:
            config = self.get_dataset_config(card)

            has_custom_config = False
        else:
            if not isinstance(config, self._config_kls):
                raise TypeError(
                    f"`config` must be of type `{self._config_kls}`, but is of type `{type(config)}` instead."
                )

            has_custom_config = True

        try:
            return self._opener(name, config)
        except ValueError as ex:
            if has_custom_config:
                raise

            msg = f"dataset_config field of the {name} asset card is not a valid {self._family} dataset configuration."

            raise AssetCardError(name, msg) from ex
        except DatasetOpenError as ex:
            msg = f"Dataset of the {name} asset card cannot be loaded."

            raise AssetCardError(name, msg) from ex
        except OSError as ex:
            raise_operational_system_error(ex)

    @override
    def open_custom_dataset(self, name: str, config: object) -> object:
        if not isinstance(config, self._config_kls):
            raise TypeError(
                f"`config` must be of type `{self._config_kls}`, but is of type `{type(config)}` instead."
            )

        try:
            return self._opener(name, config)
        except OSError as ex:
            raise_operational_system_error(ex)

    @property
    @override
    def family(self) -> str:
        return self._family

    @property
    @override
    def kls(self) -> type[object]:
        return self._kls

    @property
    @override
    def config_kls(self) -> type[object]:
        return self._config_kls


class AdvancedDatasetOpener(Protocol[DatasetConfigT_contra]):
    def __call__(
        self, resolver: DependencyResolver, name: str, config: DatasetConfigT_contra
    ) -> object: ...


def register_dataset_family(
    container: DependencyContainer,
    family: str,
    kls: type[object],
    config_kls: type[DatasetConfigT],
    *,
    opener: DatasetOpener[DatasetConfigT] | None = None,
    advanced_opener: AdvancedDatasetOpener[DatasetConfigT] | None = None,
) -> None:
    def create_handler(resolver: DependencyResolver) -> DatasetFamilyHandler:
        nonlocal opener

        if advanced_opener is not None:
            if opener is not None:
                raise ValueError(
                    "`opener` and `advanced_opener` must not be specified at the same time."
                )

            def open_dataset(name: str, config: DatasetConfigT) -> object:
                return advanced_opener(resolver, name, config)

            opener = open_dataset
        elif opener is None:
            raise ValueError("`opener` or `advanced_opener` must be specified.")

        asset_config_loader = resolver.resolve(AssetConfigLoader)

        return StandardDatasetFamilyHandler(
            family, kls, config_kls, opener, asset_config_loader
        )

    container.register(DatasetFamilyHandler, create_handler, key=family)
