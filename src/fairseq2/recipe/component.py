# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from fairseq2.dependency import DependencyContainer, DependencyResolver
from fairseq2.error import AlreadyExistsError
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate

T_co = TypeVar("T_co", bound=Hashable, covariant=True)

ConfigT_contra = TypeVar("ConfigT_contra", contravariant=True)


class ComponentFactory(Protocol[T_co, ConfigT_contra]):
    def __call__(
        self, resolver: DependencyResolver, config: ConfigT_contra
    ) -> T_co: ...


T = TypeVar("T", bound=Hashable)


@dataclass
class _ComponentEntry(Generic[T]):
    config_kls: type[Any]
    factory: ComponentFactory[T, Any]


ConfigT = TypeVar("ConfigT")


def register_component(
    container: DependencyContainer,
    kls: type[T],
    name: str,
    config_kls: type[ConfigT],
    factory: ComponentFactory[T, ConfigT],
) -> None:
    entry = _ComponentEntry[T](config_kls, factory)

    try:
        container.register_instance(_ComponentEntry, entry, key=name, kls_args=(kls,))
    except AlreadyExistsError:
        raise AlreadyExistsError(
            f"An implementation of `{kls}` with the name '{name}' is already registered."
        ) from None


def resolve_component(
    resolver: DependencyResolver, kls: type[T], name: str, config: object
) -> T:
    entry: _ComponentEntry[T]

    try:
        entry = resolver.resolve(_ComponentEntry, key=name, kls_args=(kls,))
    except LookupError:
        raise UnknownComponentError(kls, name) from None

    config = structure(config, entry.config_kls)

    validate(config)

    return entry.factory(resolver, config)


class UnknownComponentError(Exception):
    kls: type[object]
    name: str

    def __init__(self, kls: type[object], name: str) -> None:
        super().__init__(f"'{name}' is not a known `{kls}`.")

        self.kls = kls
        self.name = name
