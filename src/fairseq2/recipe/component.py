# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, cast, final

from typing_extensions import override

from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyNotFoundError,
    DependencyResolver,
)
from fairseq2.utils.structured import ValueConverter

T_co = TypeVar("T_co", covariant=True)

ConfigT_contra = TypeVar("ConfigT_contra", contravariant=True)


class ComponentFactory(Protocol[T_co, ConfigT_contra]):
    def __call__(
        self, resolver: DependencyResolver, config: ConfigT_contra
    ) -> T_co: ...


T = TypeVar("T")

ConfigT = TypeVar("ConfigT")


@dataclass
class _ComponentEntry:
    config_kls: type[Any]
    factory: ComponentFactory[Any, Any]


def register_component(
    container: DependencyContainer,
    kls: type[T],
    name: str,
    config_kls: type[ConfigT],
    factory: ComponentFactory[T, ConfigT],
) -> None:
    entry = _ComponentEntry(config_kls, factory)

    container.register_instance(_ComponentEntry, entry, key=(kls, name))


class ComponentManager(ABC):
    @abstractmethod
    def create_component(self, kls: type[T], name: str, config: object) -> T: ...

    @abstractmethod
    def structure_component_config(
        self, kls: type[object], name: str, config: object
    ) -> object: ...


@final
class _StandardComponentManager(ComponentManager):
    def __init__(
        self, resolver: DependencyResolver, value_converter: ValueConverter
    ) -> None:
        self._resolver = resolver
        self._value_converter = value_converter

    @override
    def create_component(self, kls: type[T], name: str, config: object) -> T:
        key = (kls, name)

        try:
            entry = self._resolver.resolve(_ComponentEntry, key=key)
        except DependencyNotFoundError as ex:
            if ex.kls is _ComponentEntry and ex.key == key:
                raise ComponentNotKnownError(name, kls) from None

            raise

        if not isinstance(config, entry.config_kls):
            raise TypeError(
                f"`config` must be of type `{entry.config_kls}`, but is of type `{type(config)}` instead."
            )

        component = entry.factory(self._resolver, config)

        return cast(T, component)

    @override
    def structure_component_config(
        self, kls: type[object], name: str, config: object
    ) -> object:
        key = (kls, name)

        try:
            entry = self._resolver.resolve(_ComponentEntry, key=key)
        except DependencyNotFoundError as ex:
            if ex.kls is _ComponentEntry and ex.key == key:
                raise ComponentNotKnownError(name, kls) from None

            raise

        return self._value_converter.structure(config, entry.config_kls)


class ComponentNotKnownError(Exception):
    def __init__(self, name: str, component_kls: type[object]) -> None:
        super().__init__(f"{name} is not a known `{component_kls}`.")

        self.name = name
        self.component_kls = component_kls
