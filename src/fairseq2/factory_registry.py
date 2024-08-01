# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    cast,
    final,
)

from typing_extensions import Concatenate, ParamSpec

from fairseq2.config_registry import ConfigRegistry
from fairseq2.typing import DataClass

ConfigT = TypeVar("ConfigT", bound=DataClass)

P = ParamSpec("P")

R = TypeVar("R")

R_co = TypeVar("R_co", covariant=True)


class ConfigBoundFactory(Protocol[P, R_co]):
    """Constructs instances of ``R`` using :attr:`config`."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R_co:
        ...

    @property
    def config(self) -> DataClass:
        """The configuration bound to the factory."""


@final
class ConfigBoundFactoryRegistry(Generic[P, R]):
    """Holds factories with parameter(s) ``P`` and return type ``R``."""

    _factories: Dict[
        str, Tuple[Callable[..., R], Type[DataClass], Optional[ConfigRegistry[Any]]]
    ]

    def __init__(self) -> None:
        self._factories = {}

    def get(
        self,
        name: str,
        config: Optional[DataClass] = None,
        base_config_name: Optional[str] = None,
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
                f"`name` must be a registered name, but is '{name}' instead."
            ) from None

        if config is not None:
            if not isinstance(config, config_kls):
                raise ValueError(
                    f"`config` must be of type `{config}` for '{name}', but is of type `{type(config)}` instead."
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
                config = config_registry.get(base_config_name, overwrite=config)
            except ValueError:
                raise ValueError(
                    f"`base_config_name` must be a registered configuration name, but is '{base_config_name}' instead."
                ) from None

        f = partial(factory, config)

        f.config = config  # type: ignore[attr-defined]

        return cast(ConfigBoundFactory[P, R], f)

    def register(
        self,
        name: str,
        factory: Callable[Concatenate[ConfigT, P], R],
        config_kls: Type[ConfigT],
        config_registry: Optional[ConfigRegistry[ConfigT]] = None,
    ) -> None:
        """Register ``factory`` with ``name``."""
        if name in self._factories:
            raise ValueError(
                f"`name` must be a unique name, but '{name}' is already registered."
            )

        self._factories[name] = (factory, config_kls, config_registry)
