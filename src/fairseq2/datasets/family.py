# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Final, Protocol, TypeVar, final

from typing_extensions import override

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetCardNotValidError,
    AssetConfigLoader,
)
from fairseq2.error import InternalError, raise_operational_system_error
from fairseq2.runtime.dependency import DependencyLookup, get_dependency_resolver
from fairseq2.runtime.lookup import Lookup
from fairseq2.utils.validation import ObjectValidator, ValidationError


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


def get_dataset_family(card: AssetCard) -> DatasetFamily:
    """
    Returns the :class:`DatasetFamily` for the dataset in the specified card.

    :raises AssetCardError: The card is erroneous and cannot be read.

    :raises AssetCardNotValidError: The card is missing a dataset definition
        (i.e.  `dataset_family` field).

    :raises DatasetFamilyNotKnownError: The family of the dataset is not known,
        meaning has no registered :class:`DatasetFamily`.
    """
    family = maybe_get_dataset_family(card)
    if family is None:
        message = f"{card.name} asset card is missing a dataset definition (i.e. `dataset_family` field)."

        raise AssetCardNotValidError(card.name, message)

    return family


def maybe_get_dataset_family(card: AssetCard) -> DatasetFamily | None:
    """
    Returns the :class:`DatasetFamily` for the dataset in the specified card, if
    one is defined; otherwise, returns ``None``.

    :raises AssetCardError: The card is erroneous and cannot be read.

    :raises DatasetFamilyNotKnownError: The family of the dataset is not known,
        meaning has no registered :class:`DatasetFamily`.
    """
    resolver = get_dependency_resolver()

    families = DependencyLookup(resolver, DatasetFamily)

    return _maybe_get_dataset_family(card, families)


def _maybe_get_dataset_family(
    card: AssetCard, families: Lookup[DatasetFamily]
) -> DatasetFamily | None:
    field = card.maybe_get_field("dataset_family")
    if field is None:
        return None

    family_name = field.as_(str)

    family = families.maybe_get(family_name)
    if family is None:
        raise DatasetFamilyNotKnownError(family_name)

    return family


class DatasetFamilyNotKnownError(Exception):
    """Raised when a requested dataset family is not registered."""

    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is not a known dataset family.")

        self.name = name


DatasetT_co = TypeVar("DatasetT_co", covariant=True)

DatasetConfigT_contra = TypeVar("DatasetConfigT_contra", contravariant=True)


class DatasetOpener(Protocol[DatasetConfigT_contra, DatasetT_co]):
    def __call__(self, config: DatasetConfigT_contra) -> DatasetT_co: ...


DatasetT = TypeVar("DatasetT")

DatasetConfigT = TypeVar("DatasetConfigT")


@final
class StandardDatasetFamily(DatasetFamily):
    _CONFIG_KEYS: Final = (
        "dataset_config_overrides",
        "dataset_config_override",
        "dataset_config",
    )

    def __init__(
        self,
        name: str,
        kls: type[DatasetT],
        config_kls: type[DatasetConfigT],
        opener: DatasetOpener[DatasetConfigT, DatasetT],
        validator: ObjectValidator,
        asset_config_loader: AssetConfigLoader,
    ) -> None:
        self._name = name
        self._kls: type[object] = kls
        self._config_kls: type[object] = config_kls
        self._opener: DatasetOpener[Any, object] = opener
        self._validator = validator
        self._asset_config_loader = asset_config_loader

    @override
    def get_dataset_config(self, card: AssetCard) -> object:
        try:
            base_config = self._config_kls()
        except TypeError as ex:
            raise InternalError(
                f"Default configuration of the {self._name} dataset family cannot be constructed."
            ) from ex

        name = card.name

        for key in self._CONFIG_KEYS:
            config = self._asset_config_loader.load(card, base_config, config_key=key)

            if config is not base_config:
                try:
                    self._validator.validate(config)
                except ValidationError as ex:
                    msg = f"{key} field of the {name} asset card is not a valid {self._name} dataset configuration."

                    raise AssetCardError(name, msg) from ex

                return config

        return base_config

    @override
    def open_dataset(self, card: AssetCard, config: object | None) -> object:
        if config is None:
            config = self.get_dataset_config(card)
        else:
            if not isinstance(config, self._config_kls):
                raise TypeError(
                    f"`config` must be of type `{self._config_kls}`, but is of type `{type(config)}` instead."
                )

        name = card.name

        try:
            return self._opener(config)
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
