# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Set
from functools import cached_property
from typing import Any, Generic, Protocol, TypeVar, final, get_args

from typing_extensions import override

from fairseq2.error import AlreadyExistsError

ConfigT = TypeVar("ConfigT")

ConfigT_co = TypeVar("ConfigT_co", covariant=True)


class ConfigProvider(ABC, Generic[ConfigT_co]):
    """Provides configurations of type ``ConfigT``."""

    @abstractmethod
    def get(self, name: str) -> ConfigT_co:
        """Return the configuration of ``name``."""

    @abstractmethod
    def names(self) -> Set[str]:
        """Return the names of all configurations."""


class ConfigSupplier(Protocol[ConfigT_co]):
    """Supplies instances of ``ConfigT``."""

    def __call__(self) -> ConfigT_co:
        ...


@final
class ConfigRegistry(ConfigProvider[ConfigT]):
    """Holds configurations of type ``ConfigT``."""

    _configs: dict[str, ConfigSupplier[ConfigT]]

    def __init__(self) -> None:
        self._configs = {}

    @override
    def get(self, name: str) -> ConfigT:
        """Return the configuration of ``name``."""
        try:
            return self._configs[name]()
        except KeyError:
            raise LookupError(
                f"`name` must be a registered configuration name, but '{name}' is not registered."
            ) from None

    def register(self, name: str, supplier: ConfigSupplier[ConfigT]) -> None:
        """Register a new configuration.

        :param name:
            The name of the configuration.
        :param config_supplier:
            The supplier to retrieve configurations.
        """
        if name in self._configs:
            raise AlreadyExistsError(
                f"`name` must be a unique configuration name, but '{name}' is already registered."
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
        """Return the names of all configurations."""
        return self._configs.keys()

    @cached_property
    def config_kls(self) -> Any:
        kls_args = get_args(self.__orig_class__)  # type: ignore[attr-defined]

        return kls_args[0]
