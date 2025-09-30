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
from fairseq2.runtime.lookup import Lookup


class DatasetFamily(ABC):
    @abstractmethod
    def get_dataset_config(self, card: AssetCard) -> object: ...

    @abstractmethod
    def open_dataset(self, card: AssetCard, config: object | None) -> object: ...

    @abstractmethod
    def open_custom_dataset(self, config: object) -> object: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def kls(self) -> type[object]: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...


class DatasetError(Exception):
    pass


def get_dataset_family(
    card: AssetCard, families: Lookup[DatasetFamily]
) -> DatasetFamily:
    family_name = card.field("dataset_family").as_(str)

    family = families.maybe_get(family_name)
    if family is None:
        msg = f"family field of the {card.name} asset card is expected to be a supported dataset family, but is {family_name} instead."

        raise AssetCardError(card.name, msg)

    return family


DatasetT_co = TypeVar("DatasetT_co", covariant=True)

DatasetConfigT_contra = TypeVar("DatasetConfigT_contra", contravariant=True)


class DatasetOpener(Protocol[DatasetConfigT_contra, DatasetT_co]):
    def __call__(self, config: DatasetConfigT_contra) -> DatasetT_co: ...


DatasetT = TypeVar("DatasetT")

DatasetConfigT = TypeVar("DatasetConfigT")


@final
class StandardDatasetFamily(DatasetFamily):
    def __init__(
        self,
        name: str,
        kls: type[DatasetT],
        config_kls: type[DatasetConfigT],
        opener: DatasetOpener[DatasetConfigT, DatasetT],
        asset_config_loader: AssetConfigLoader,
    ) -> None:
        self._name = name
        self._kls: type[object] = kls
        self._config_kls: type[object] = config_kls
        self._opener: DatasetOpener[Any, object] = opener
        self._asset_config_loader = asset_config_loader

    @override
    def get_dataset_config(self, card: AssetCard) -> object:
        try:
            base_config = self._config_kls()
        except TypeError as ex:
            raise InternalError(
                f"Default configuration of the {self._name} dataset family cannot be constructed."
            ) from ex

        # legacy
        base_config = self._asset_config_loader.load(
            card, base_config, config_key="dataset_config"
        )

        base_config = self._asset_config_loader.load(
            card, base_config, config_key="dataset_config_override"
        )

        return base_config

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
            return self._opener(config)
        except ValueError as ex:
            if has_custom_config:
                raise

            msg = f"dataset_config field of the {name} asset card is not a valid {self._name} dataset configuration."

            raise AssetCardError(name, msg) from ex
        except DatasetError as ex:
            msg = f"Dataset of the {name} asset card cannot be opened."

            raise AssetCardError(name, msg) from ex
        except OSError as ex:
            raise_operational_system_error(ex)

    @override
    def open_custom_dataset(self, config: object) -> object:
        if not isinstance(config, self._config_kls):
            raise TypeError(
                f"`config` must be of type `{self._config_kls}`, but is of type `{type(config)}` instead."
            )

        try:
            return self._opener(config)
        except OSError as ex:
            raise_operational_system_error(ex)

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def kls(self) -> type[object]:
        return self._kls

    @property
    @override
    def config_kls(self) -> type[object]:
        return self._config_kls
