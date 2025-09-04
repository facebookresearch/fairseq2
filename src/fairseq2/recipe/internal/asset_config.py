# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final

from typing_extensions import override

from fairseq2.error import InternalError
from fairseq2.recipe.error import RecipeConfigParseError
from fairseq2.utils.config import ConfigDirectiveError, ConfigMerger, ConfigProcessor
from fairseq2.utils.structured import StructureError, ValueConverter
from fairseq2.utils.validation import ObjectValidator, ValidationError


class _AssetConfigOverrider(ABC):
    @abstractmethod
    def apply_overrides(
        self, section_name: str, config: object, unstructured_overrides: object
    ) -> object: ...


@final
class _StandardAssetConfigOverrider(_AssetConfigOverrider):
    def __init__(
        self,
        value_converter: ValueConverter,
        config_merger: ConfigMerger,
        config_processor: ConfigProcessor,
        validator: ObjectValidator,
    ) -> None:
        self._value_converter = value_converter
        self._config_merger = config_merger
        self._config_processor = config_processor
        self._validator = validator

    @override
    def apply_overrides(
        self, section_name: str, config: object, config_overrides: object
    ) -> object:
        if config_overrides is None:
            return config

        # TODO(balioglu): unescape _set_ and _del_ in config_overrides
        try:
            unstructured_config = self._value_converter.unstructure(config)
        except StructureError as ex:
            raise InternalError("`config` cannot be unstructured") from ex

        try:
            unstructured_config = self._config_merger.merge(
                unstructured_config, config_overrides
            )
        except (ValueError, TypeError) as ex:
            raise RecipeConfigParseError(
                f"`{section_name}.config_overrides` cannot be merged with the base configuration."
            ) from ex

        # TODO(balioglu): unescape config directives and run them.
        try:
            unstructured_config = self._config_processor.process(unstructured_config)
        except ConfigDirectiveError as ex:
            raise RecipeConfigParseError(
                f"A directive in `{section_name}.config_overrides` cannot be processed."
            ) from ex

        config_kls = type(config)

        try:
            config = self._value_converter.structure(unstructured_config, config_kls)
        except StructureError as ex:
            raise RecipeConfigParseError(
                f"`{section_name}.config_overrides` cannot be structured."
            ) from ex

        try:
            self._validator.validate(config)
        except ValidationError as ex:
            raise ValidationError(
                ex.result, field=f"{section_name}.config_overrides"
            ) from None

        return config
