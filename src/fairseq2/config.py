# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, Literal, Protocol, TypeAlias, TypeVar, final, overload

from fairseq2.dependency import DependencyContainer, DependencyResolver

ConfigT_co = TypeVar("ConfigT_co", covariant=True)


class ConfigSupplier(Protocol[ConfigT_co]):
    def __call__(self) -> ConfigT_co: ...


class ContextBackedConfigSupplier(Protocol[ConfigT_co]):
    def __call__(self, resolver: DependencyResolver) -> ConfigT_co: ...


ConfigT = TypeVar("ConfigT")

ConfigDecorator: TypeAlias = Callable[
    [ConfigSupplier[ConfigT]], ConfigSupplier[ConfigT]
]

ContextBackedConfigDecorator: TypeAlias = Callable[
    [ContextBackedConfigSupplier[ConfigT]], ContextBackedConfigSupplier[ConfigT]
]


@final
class ConfigRegistrar(Generic[ConfigT]):
    _container: DependencyContainer
    _config_kls: type[ConfigT]

    def __init__(
        self, container: DependencyContainer, config_kls: type[ConfigT]
    ) -> None:
        self._container = container
        self._config_kls = config_kls

    @overload
    def __call__(
        self, name: str, *, resolver: Literal[True]
    ) -> ContextBackedConfigDecorator[ConfigT]: ...

    @overload
    def __call__(
        self, name: str, *, resolver: Literal[False]
    ) -> ConfigDecorator[ConfigT]: ...

    @overload
    def __call__(self, name: str) -> ConfigDecorator[ConfigT]: ...

    @overload
    def __call__(
        self, name: str, *, resolver: bool = ...
    ) -> ConfigDecorator[ConfigT] | ContextBackedConfigDecorator[ConfigT]: ...

    def __call__(
        self, name: str, resolver: bool = False
    ) -> ConfigDecorator[ConfigT] | ContextBackedConfigDecorator[ConfigT]:
        if resolver:

            def context_backed_decorator(
                supplier: ContextBackedConfigSupplier[ConfigT],
            ) -> ContextBackedConfigSupplier[ConfigT]:
                def get_config(resolver: DependencyResolver) -> ConfigT:
                    return supplier(resolver)

                self._container.register(self._config_kls, get_config, key=name)

                self._container.register_instance(str, name, key=self._config_kls)  # type: ignore[arg-type]

                return supplier

            return context_backed_decorator
        else:

            def decorator(supplier: ConfigSupplier[ConfigT]) -> ConfigSupplier[ConfigT]:
                def get_config(resolver: DependencyResolver) -> ConfigT:
                    return supplier()

                self._container.register(self._config_kls, get_config, key=name)

                self._container.register_instance(str, name, key=self._config_kls)  # type: ignore[arg-type]

                return supplier

            return decorator


def get_config(
    resolver: DependencyResolver, config_kls: type[ConfigT], name: str
) -> ConfigT:
    return resolver.resolve(config_kls, key=name)


def get_config_names(
    resolver: DependencyResolver, config_kls: type[object]
) -> list[str]:
    it = resolver.resolve(str, key=config_kls)  # type: ignore[arg-type]

    return list(it)
