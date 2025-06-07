# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import (
    Generic,
    Iterable,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    final,
    overload,
)

from typing_extensions import override

from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyNotFoundError,
    DependencyResolver,
)

ConfigT_co = TypeVar("ConfigT_co", covariant=True)


class BasicConfigSupplier(Protocol[ConfigT_co]):
    def __call__(self) -> ConfigT_co: ...


class ConfigSupplier(Protocol[ConfigT_co]):
    def __call__(self, resolver: DependencyResolver) -> ConfigT_co: ...


ConfigT = TypeVar("ConfigT")

BasicConfigDecorator: TypeAlias = Callable[
    [BasicConfigSupplier[ConfigT]], BasicConfigSupplier[ConfigT]
]

ConfigDecorator: TypeAlias = Callable[
    [ConfigSupplier[ConfigT]], ConfigSupplier[ConfigT]
]


@final
class ConfigRegistrar(Generic[ConfigT]):
    _container: DependencyContainer
    _kls: type[ConfigT]

    def __init__(self, container: DependencyContainer, kls: type[ConfigT]) -> None:
        self._container = container
        self._kls = kls

    @overload
    def __call__(self, name: str) -> BasicConfigDecorator[ConfigT]: ...

    @overload
    def __call__(
        self, name: str, *, resolver: Literal[False]
    ) -> BasicConfigDecorator[ConfigT]: ...

    @overload
    def __call__(
        self, name: str, *, resolver: Literal[True]
    ) -> ConfigDecorator[ConfigT]: ...

    @overload
    def __call__(
        self, name: str, *, resolver: bool = ...
    ) -> BasicConfigDecorator[ConfigT] | ConfigDecorator[ConfigT]: ...

    def __call__(
        self, name: str, resolver: bool = False
    ) -> BasicConfigDecorator[ConfigT] | ConfigDecorator[ConfigT]:
        if resolver:

            def decorator(supplier: ConfigSupplier[ConfigT]) -> ConfigSupplier[ConfigT]:
                def get_config(resolver: DependencyResolver) -> ConfigT:
                    return supplier(resolver)

                self._container.register(self._kls, get_config, key=name)

                self._container.register_instance(str, name, key=self._kls)  # type: ignore[arg-type]

                return supplier

            return decorator
        else:

            def basic_decorator(
                supplier: BasicConfigSupplier[ConfigT],
            ) -> BasicConfigSupplier[ConfigT]:
                def get_config(resolver: DependencyResolver) -> ConfigT:
                    return supplier()

                self._container.register(self._kls, get_config, key=name)

                self._container.register_instance(str, name, key=self._kls)  # type: ignore[arg-type]

                return supplier

            return basic_decorator


class ConfigProvider(ABC, Generic[ConfigT_co]):
    @abstractmethod
    def get_config(self, name: str) -> ConfigT_co: ...

    @abstractmethod
    def get_config_names(self) -> Iterable[str]: ...

    @property
    @abstractmethod
    def kls(self) -> type[ConfigT_co]: ...


@final
class ResolverBackedConfigProvider(ConfigProvider[ConfigT]):
    _resolver: DependencyResolver
    _kls: type[ConfigT]

    def __init__(self, resolver: DependencyResolver, kls: type[ConfigT]) -> None:
        self._resolver = resolver
        self._kls = kls

    @override
    def get_config(self, name: str) -> ConfigT:
        try:
            return self._resolver.resolve(self._kls, key=name)
        except DependencyNotFoundError:
            raise ConfigNotFoundError(self._kls, name) from None

    @override
    def get_config_names(self) -> Iterable[str]:
        return self._resolver.resolve(str, key=self._kls)  # type: ignore[arg-type]

    @property
    @override
    def kls(self) -> type[ConfigT]:
        return self._kls


class ConfigNotFoundError(LookupError):
    kls: type[object]
    name: str

    def __init__(self, kls: type[object], name: str) -> None:
        super().__init__(f"A configuration named '{name}' cannot be found for `{kls}`.")

        self.kls = kls
        self.name = name


def get_config(resolver: DependencyResolver, kls: type[ConfigT], name: str) -> ConfigT:
    try:
        return resolver.resolve(kls, key=name)
    except DependencyNotFoundError:
        raise ConfigNotFoundError(kls, name) from None
