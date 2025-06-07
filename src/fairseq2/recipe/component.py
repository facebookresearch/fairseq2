# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from fairseq2.dependency import DependencyContainer, DependencyResolver
from fairseq2.error import AlreadyExistsError
from fairseq2.utils.structured import StructureError, structure
from fairseq2.utils.validation import validate

ConfigT_contra = TypeVar("ConfigT_contra", contravariant=True)


class ComponentFactory(Protocol[ConfigT_contra]):
    def __call__(
        self, resolver: DependencyResolver, config: ConfigT_contra
    ) -> object: ...


@dataclass
class _ComponentEntry:
    config_kls: type[Any]
    factory: ComponentFactory[Any]


ConfigT = TypeVar("ConfigT")


def register_component(
    container: DependencyContainer,
    name: str,
    config_kls: type[ConfigT],
    factory: ComponentFactory[ConfigT],
) -> None:
    entry = _ComponentEntry(config_kls, factory)

    try:
        container.register_instance(_ComponentEntry, entry, key=name)
    except AlreadyExistsError:
        raise AlreadyExistsError(
            f"A component named '{name}' is already registered."
        ) from None


T = TypeVar("T")


def resolve_component(
    resolver: DependencyResolver, kls: type[T], name: str, config: object
) -> T:
    try:
        entry = resolver.resolve(_ComponentEntry, key=name)
    except LookupError:
        raise UnknownComponentError(name) from None

    try:
        config = structure(config, entry.config_kls)
    except StructureError as ex:
        raise StructureError(
            f"The '{name}' component cannot be structured. See the nested exception for details."
        ) from ex

    validate(config)

    component = entry.factory(resolver, config)

    if not isinstance(component, kls):
        raise TypeError(
            f"The '{name}' component must be of type `{kls}`, but is of type `{type(component)}` instead."
        )

    return component


class UnknownComponentError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is not a known component.")

        self.name = name
