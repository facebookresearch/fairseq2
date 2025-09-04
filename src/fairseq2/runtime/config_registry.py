# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import (
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    final,
    overload,
)

from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyNotFoundError,
    DependencyResolver,
)

ConfigT_co = TypeVar("ConfigT_co", covariant=True)


class ConfigProvider(Protocol[ConfigT_co]):
    def __call__(self) -> ConfigT_co: ...


class AdvancedConfigProvider(Protocol[ConfigT_co]):
    def __call__(self, resolver: DependencyResolver) -> ConfigT_co: ...


ConfigT = TypeVar("ConfigT")

ConfigDecorator: TypeAlias = Callable[
    [ConfigProvider[ConfigT]], ConfigProvider[ConfigT]
]

AdvancedConfigDecorator: TypeAlias = Callable[
    [AdvancedConfigProvider[ConfigT]], AdvancedConfigProvider[ConfigT]
]


@final
class ConfigRegistrar(Generic[ConfigT]):
    def __init__(self, container: DependencyContainer, kls: type[ConfigT]) -> None:
        self._container = container
        self._kls = kls

    @overload
    def __call__(self, name: str) -> ConfigDecorator[ConfigT]: ...

    @overload
    def __call__(
        self, name: str, *, advanced: Literal[False]
    ) -> ConfigDecorator[ConfigT]: ...

    @overload
    def __call__(
        self, name: str, *, advanced: Literal[True]
    ) -> AdvancedConfigDecorator[ConfigT]: ...

    @overload
    def __call__(
        self, name: str, *, advanced: bool = ...
    ) -> ConfigDecorator[ConfigT] | AdvancedConfigDecorator[ConfigT]: ...

    def __call__(
        self, name: str, advanced: bool = False
    ) -> ConfigDecorator[ConfigT] | AdvancedConfigDecorator[ConfigT]:
        if advanced:

            def advanced_decorator(
                provider: AdvancedConfigProvider[ConfigT],
            ) -> AdvancedConfigProvider[ConfigT]:
                def get_config(resolver: DependencyResolver) -> ConfigT:
                    return provider(resolver)

                self._container.register(self._kls, get_config, key=name)

                return provider

            return advanced_decorator
        else:

            def decorator(provider: ConfigProvider[ConfigT]) -> ConfigProvider[ConfigT]:
                def get_config(resolver: DependencyResolver) -> ConfigT:
                    return provider()

                self._container.register(self._kls, get_config, key=name)

                return provider

            return decorator


def get_config(resolver: DependencyResolver, kls: type[ConfigT], name: str) -> ConfigT:
    try:
        return resolver.resolve(kls, key=name)
    except DependencyNotFoundError as ex:
        if ex.kls is kls and ex.key == name:
            raise ConfigNotFoundError(name) from None

        raise


class ConfigNotFoundError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"A configuration named {name} cannot be found.")

        self.name = name
