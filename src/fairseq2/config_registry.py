# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Set
from typing import Generic, Protocol, TypeVar, final

from typing_extensions import override

from fairseq2.error import AlreadyExistsError

ConfigT_co = TypeVar("ConfigT_co", covariant=True)


class ConfigProvider(ABC, Generic[ConfigT_co]):
    """Provides configurations of type ``ConfigT``."""

    @abstractmethod
    def get(self, name: str) -> ConfigT_co:
        """Return the configuration of ``name``."""

    @abstractmethod
    def has(self, name: str) -> bool: ...

    @abstractmethod
    def names(self) -> Set[str]:
        """Return the names of all configurations."""

    @property
    @abstractmethod
    def config_kls(self) -> type[ConfigT_co]:
        """The type of the configuration."""


class ConfigSupplier(Protocol[ConfigT_co]):
    def __call__(self) -> ConfigT_co: ...


ConfigT = TypeVar("ConfigT")


@final
class ConfigRegistry(ConfigProvider[ConfigT]):
    """Holds configurations of type ``ConfigT``."""

    _configs: dict[str, ConfigSupplier[ConfigT]]
    _config_kls: type[ConfigT]

    def __init__(self, config_kls: type[ConfigT]) -> None:
        self._configs = {}
        self._config_kls = config_kls

    @override
    def get(self, name: str) -> ConfigT:
        try:
            return self._configs[name]()
        except KeyError:
            raise ConfigNotFoundError(name) from None

    @override
    def has(self, name: str) -> bool:
        return name in self._configs

    def register(self, name: str, supplier: ConfigSupplier[ConfigT]) -> None:
        """Register a new configuration.

        :param name: The name of the configuration.
        :param config_supplier: The configuration supplier.
        """
        if name in self._configs:
            raise AlreadyExistsError(
                f"The registry has already a configuration named '{name}'."
            )

        self._configs[name] = supplier

    def decorator(
        self, name: str
    ) -> Callable[[ConfigSupplier[ConfigT]], ConfigSupplier[ConfigT]]:
        """Register ``name`` with the decorated configuration supplier."""

        def register(supplier: ConfigSupplier[ConfigT]) -> ConfigSupplier[ConfigT]:
            self.register(name, supplier)

            return supplier

        return register

    @override
    def names(self) -> Set[str]:
        return self._configs.keys()

    @property
    @override
    def config_kls(self) -> type[ConfigT]:
        return self._config_kls


class ConfigNotFoundError(LookupError):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a registered configuration name.")

        self.name = name
