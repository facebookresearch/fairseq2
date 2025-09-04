# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar, final

from typing_extensions import override

from fairseq2.utils.config import ConfigDirectiveError, ConfigMerger, ConfigProcessor
from fairseq2.utils.structured import StructureError, ValueConverter
from fairseq2.utils.uri import Uri

T = TypeVar("T", bool, int, float, str)


@final
class AssetCard:
    """Holds information about an asset."""

    def __init__(
        self, name: str, metadata: Mapping[str, object], base: AssetCard | None = None
    ) -> None:
        self._name = name
        self._metadata = metadata
        self._base = base

    def field(self, name: str) -> AssetCardField:
        field = self.maybe_get_field(name)
        if field is None:
            msg = f"{self._name} asset card does not have a field named {name}."

            raise AssetCardError(self._name, msg)

        return field

    def maybe_get_field(self, name: str) -> AssetCardField | None:
        card: AssetCard | None = self

        while card is not None:
            try:
                value = card.metadata[name]
            except KeyError:
                pass
            else:
                return AssetCardField(name, self, value)

            card = card.base

        return None

    def has_field(self, name: str) -> bool:
        card: AssetCard | None = self

        while card is not None:
            if name in card.metadata:
                return True

            card = card.base

        return False

    def __repr__(self) -> str:
        return f"{self._name}={self._metadata}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def metadata(self) -> Mapping[str, object]:
        return self._metadata

    @property
    def base(self) -> AssetCard | None:
        return self._base


@final
class AssetCardField:
    def __init__(self, name: str, card: AssetCard, value: object) -> None:
        self._name = name
        self._card = card
        self._value = value

    @property
    def value(self) -> object:
        return self._value

    def as_(self, kls: type[T]) -> T:
        if not isinstance(self._value, kls):
            msg = f"{self._name} field of the {self._card.name} asset card is expected to be of type `{kls}`, but is of type `{type(self._value)}` instead."

            raise AssetCardError(self._card.name, msg)

        return self._value

    def as_uri(self) -> Uri:
        value = self.as_(str)

        uri = Uri.maybe_parse(value)
        if uri is not None:
            return uri

        try:
            path = Path(value)
        except ValueError:
            msg = f"{self._name} field of the {self._card.name} asset card cannot be parsed as a URI or a pathname."

            raise AssetCardError(self._card.name, msg) from None

        if not path.is_absolute():
            base_path = self._card.metadata.get("__base_path__")
            if not isinstance(base_path, Path):
                msg = f"{self._name} field of the {self._card.name} asset card is a relative pathname and cannot be converted to a URI."

                raise AssetCardError(self._card.name, msg)

            path = base_path.joinpath(path)

        return Uri.from_path(path)


class AssetCardError(Exception):
    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)

        self.name = name


class AssetConfigLoader(ABC):
    @abstractmethod
    def load(self, card: AssetCard, base_config: object, config_key: str) -> object: ...


@final
class StandardAssetConfigLoader(AssetConfigLoader):
    def __init__(
        self,
        value_converter: ValueConverter,
        config_merger: ConfigMerger,
        config_processor: ConfigProcessor,
    ) -> None:
        self._value_converter = value_converter
        self._config_merger = config_merger
        self._config_processor = config_processor

    @override
    def load(self, card: AssetCard, base_config: object, config_key: str) -> object:
        all_config_overrides = []

        base_card: AssetCard | None = card

        while base_card is not None:
            config_overrides = base_card.metadata.get(config_key)
            if config_overrides is not None:
                all_config_overrides.append((base_card.name, config_overrides))

            base_card = base_card.base

        if not all_config_overrides:
            return base_config

        config_kls = type(base_config)

        try:
            unstructured_config = self._value_converter.unstructure(base_config)
        except StructureError:
            msg = f"{config_key} field of the {card.name} asset card cannot be parsed as of type `{config_kls}`."

            raise AssetCardError(card.name, msg) from None

        for name, config_overrides in reversed(all_config_overrides):
            # TODO(balioglu): unescape _set_ and _del_ in config_overrides
            try:
                unstructured_config = self._config_merger.merge(
                    unstructured_config, config_overrides
                )
            except (ValueError, TypeError) as ex:
                msg = f"{config_key} field of the {name} asset card cannot be merged with the base configuration."

                raise AssetCardError(name, msg) from ex

            # TODO(balioglu): unescape config directives and run them.
            try:
                unstructured_config = self._config_processor.process(
                    unstructured_config
                )
            except ConfigDirectiveError as ex:
                msg = f"A directive in the {config_key} field of the {name} asset card cannot processed."

                raise AssetCardError(name, msg) from ex

        try:
            return self._value_converter.structure(unstructured_config, config_kls)
        except StructureError as ex:
            msg = f"{config_key} field of the {card.name} asset card cannot be parsed as of type `{config_kls}`."

            raise AssetCardError(card.name, msg) from ex
