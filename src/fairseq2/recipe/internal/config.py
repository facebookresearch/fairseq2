# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from typing import TypeVar, final, get_type_hints

from typing_extensions import override

from fairseq2.recipe.component import ComponentManager
from fairseq2.recipe.config import ModelSection, SupportsStructure
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.typing import is_dataclass_instance
from fairseq2.utils.structured import StructureError, ValueConverter


def _is_train_config(config_kls: type[object]) -> bool:
    if is_dataclass(config_kls):
        type_hints = get_type_hints(config_kls)

        model_section_kls = type_hints.get("model")
        if model_section_kls is not None:
            return issubclass(model_section_kls, ModelSection)

    return False


@dataclass
class _RecipeConfigHolder:
    config: object


SectionT = TypeVar("SectionT")


def _get_config_section(
    resolver: DependencyResolver, name: str, kls: type[SectionT]
) -> SectionT:
    config_holder = resolver.resolve(_RecipeConfigHolder)

    try:
        section = getattr(config_holder.config, name)
    except AttributeError:
        raise LookupError() from None

    if not isinstance(section, kls):
        raise TypeError(
            f"`{name}` recipe configuration section must be of type `{kls}`, but is of type `{type(section)}` instead."
        )

    return section


class _RecipeConfigStructurer(ABC):
    @abstractmethod
    def structure(
        self, config_kls: type[object], unstructured_config: object
    ) -> object: ...


@final
class _StandardRecipeConfigStructurer(_RecipeConfigStructurer):
    def __init__(
        self,
        component_manager: ComponentManager,
        value_converter: ValueConverter,
        resolver: DependencyResolver,
    ) -> None:
        self._component_manager = component_manager
        self._value_converter = value_converter
        self._resolver = resolver

    @override
    def structure(
        self, config_kls: type[object], unstructured_config: object
    ) -> object:
        config = self._value_converter.structure(unstructured_config, config_kls)

        self._structure_sections(config)

        return config

    def _structure_sections(self, obj: object) -> None:
        if isinstance(obj, SupportsStructure):
            obj.structure(self._resolver)

        if isinstance(obj, list):
            for idx, e in enumerate(obj):
                try:
                    self._structure_sections(e)
                except StructureError as ex:
                    raise StructureError(
                        f"Element at index {idx} cannot be structured."
                    ) from ex

            return

        if isinstance(obj, dict):
            for k, v in obj.items():
                try:
                    self._structure_sections(k)
                except StructureError as ex:
                    raise StructureError(f"{k} key cannot be structured.") from ex

                try:
                    self._structure_sections(v)
                except StructureError as ex:
                    raise StructureError(
                        f"Value of the {k} key cannot be structured."
                    ) from ex

            return

        if is_dataclass_instance(obj):
            for f in fields(obj):
                v = getattr(obj, f.name)

                try:
                    self._structure_sections(v)
                except StructureError as ex:
                    raise StructureError(
                        f"`{f.name}` field cannot be structured."
                    ) from ex
