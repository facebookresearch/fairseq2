# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable, Set
from typing import Generic, Protocol, TypeVar, final

from fairseq2.typing import DataClass
from fairseq2.utils.dataclass import empty, update_dataclass

ConfigT = TypeVar("ConfigT", bound=DataClass)

ConfigT_co = TypeVar("ConfigT_co", bound=DataClass, covariant=True)


class ConfigFactory(Protocol[ConfigT_co]):
    def __call__(self) -> ConfigT_co:
        ...


@final
class ConfigRegistry(Generic[ConfigT]):
    """Holds configurations of type ``ConfigT``."""

    _configs: dict[str, ConfigFactory[ConfigT]]

    def __init__(self) -> None:
        self._configs = {}

    def get(
        self,
        name: str,
        *,
        overwrite: ConfigT | None = None,
        return_empty: bool = False,
    ) -> ConfigT:
        """Return the configuration of ``name``.

        :param overwrite:
            The configuration whose non-empty fields will overwrite the returned
            configuration.
        :param return_empty:
            If ``True``, all fields of the returned configuration will be set to
            empty (i.e. ``EMPTY``).
        """
        try:
            config = self._configs[name]()
        except KeyError:
            raise ValueError(
                f"`name` must be a registered configuration name, but is '{name}' instead."
            ) from None

        if overwrite is not None:
            update_dataclass(config, overwrite)

        if return_empty:
            empty(config)

        return config

    def register(self, name: str, config_factory: ConfigFactory[ConfigT]) -> None:
        """Register a new configuration.

        :param name:
            The name of the configuration.
        :param config_factory:
            The factory to construct configurations.
        """
        if name in self._configs:
            raise ValueError(
                f"`name` must be a unique configuration name, but '{name}' is already registered."
            )

        self._configs[name] = config_factory

    def decorator(
        self, name: str
    ) -> Callable[[ConfigFactory[ConfigT]], ConfigFactory[ConfigT]]:
        """Register ``name`` with the decorated configuration factory."""

        def register(config_factory: ConfigFactory[ConfigT]) -> ConfigFactory[ConfigT]:
            self.register(name, config_factory)

            return config_factory

        return register

    def names(self) -> Set[str]:
        """Return the names of all configurations."""
        return self._configs.keys()
