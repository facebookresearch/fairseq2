# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import is_dataclass
from functools import partial
from inspect import isfunction
from typing import (
    Any,
    Generic,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
    final,
    get_type_hints,
)

from fairseq2.config_registry import ConfigRegistry
from fairseq2.typing import DataClass
from fairseq2.utils.dataclass import merge_dataclass
from fairseq2.utils.structured import ValueConverter, default_value_converter

ConfigT = TypeVar("ConfigT", bound=DataClass)

ConfigT_contra = TypeVar("ConfigT_contra", bound=DataClass, contravariant=True)

P = ParamSpec("P")

R = TypeVar("R")

R_co = TypeVar("R_co", covariant=True)


class Factory(Protocol[ConfigT_contra, P, R_co]):
    def __call__(
        self, config: ConfigT_contra, *args: P.args, **kwargs: P.kwargs
    ) -> R_co:
        ...


class ConfigBoundFactory(Protocol[P, R_co]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R_co:
        ...

    @property
    def config(self) -> DataClass:
        ...


@final
class ConfigBoundFactoryRegistry(Generic[P, R]):
    """Holds factories with parameter(s) ``P`` and return type ``R``."""

    _factories: dict[
        str, tuple[Callable[..., R], type[DataClass], ConfigRegistry[Any] | None]
    ]

    _value_converter: ValueConverter

    def __init__(self, value_converter: ValueConverter | None = None) -> None:
        self._factories = {}

        self._value_converter = value_converter or default_value_converter

    def get(
        self,
        name: str,
        unstructured_config: object = None,
        base_config_name: str | None = None,
        *,
        set_empty: bool = False,
    ) -> ConfigBoundFactory[P, R]:
        """Return the factory with ``name``.

        :param config:
            The configuration to bind to the factory.
        :param base_config_name:
            The name of the configuration on which ``config`` will be based.
        """
        try:
            factory, config_kls, config_registry = self._factories[name]
        except KeyError:
            raise ValueError(
                f"`name` must be a registered name, but '{name}' is not registered."
            ) from None

        if unstructured_config is None:
            config = None
        else:
            config = self._value_converter.structure(
                unstructured_config, config_kls, set_empty=set_empty
            )

        if base_config_name is None:
            if config is None:
                try:
                    config = config_kls()
                except TypeError as ex:
                    raise RuntimeError(
                        f"'{name}' has no default configuration."
                    ) from ex
        else:
            if config_registry is None:
                raise ValueError(
                    f"`base_config_name` must be a registered configuration name, but is '{base_config_name}' instead."
                )

            try:
                base_config = config_registry.get(base_config_name)
            except ValueError:
                raise ValueError(
                    f"`base_config_name` must be a registered configuration name, but is '{base_config_name}' instead."
                ) from None

            if config is None:
                config = base_config
            else:
                config = merge_dataclass(base_config, config)

        f = partial(factory, config)

        f.config = config  # type: ignore[attr-defined]

        return cast(ConfigBoundFactory[P, R], f)

    def register(
        self,
        name: str,
        factory: Factory[ConfigT, P, R],
        config_kls: type[ConfigT],
        config_registry: ConfigRegistry[ConfigT] | None = None,
    ) -> None:
        """Register ``factory`` with ``name``."""
        if name in self._factories:
            raise ValueError(
                f"`name` must be a unique name, but '{name}' is already registered."
            )

        self._factories[name] = (factory, config_kls, config_registry)

    def decorator(
        self, name: str
    ) -> Callable[[Factory[ConfigT, P, R]], Factory[ConfigT, P, R]]:
        """Register ``name`` with the decorated factory function."""

        def register(factory: Factory[ConfigT, P, R]) -> Factory[ConfigT, P, R]:
            if not isfunction(factory):
                raise TypeError("`factory` must be a function.")

            type_hints = get_type_hints(factory)

            if len(type_hints) < 2:
                raise ValueError(
                    f"The decorated factory `{factory}` must have at least one parameter."
                )

            config_kls = next(iter(type_hints.values()))

            if not is_dataclass(config_kls):
                raise ValueError(
                    f"The first parameter of the decorated factory `{factory}` must be a dataclass."
                )

            self.register(name, factory, config_kls)

            return factory

        return register
