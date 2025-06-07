# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields
from typing import final

from typing_extensions import override

from fairseq2.recipe.component import ComponentManager
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.typing import is_dataclass_instance
from fairseq2.utils.structured import StructureError, ValueConverter
from fairseq2.utils.validation import ObjectValidator


@final
class RecipeConfigPreparer:
    def __init__(
        self, structurer: RecipeConfigStructurer, validator: ObjectValidator
    ) -> None:
        self._structurer = structurer
        self._validator = validator

    def prepare(self, unstructured_config: object) -> object:
        try:
            config = self._structurer.structure(unstructured_config)
        except StructureError as ex:
            raise RecipeConfigError(
                "Recipe configuration cannot be structured."
            ) from ex

        self._validator.validate(config)

        return config


class RecipeConfigError(Exception):
    pass


class RecipeConfigStructurer(ABC):
    @abstractmethod
    def structure(self, unstructured_config: object) -> object: ...


@final
class StandardRecipeConfigStructurer(RecipeConfigStructurer):
    def __init__(
        self,
        config_kls: type[object],
        component_manager: ComponentManager,
        value_converter: ValueConverter,
        resolver: DependencyResolver,
    ) -> None:
        self._config_kls = config_kls
        self._component_manager = component_manager
        self._value_converter = value_converter
        self._resolver = resolver

    @override
    def structure(self, unstructured_config: object) -> object:
        config = self._value_converter.structure(unstructured_config, self._config_kls)

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


class SupportsStructure(ABC):
    @abstractmethod
    def structure(self, resolver: DependencyResolver) -> None: ...
