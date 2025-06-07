# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar, final
from urllib.parse import urlparse, urlunparse

from fairseq2.error import ContractError
from fairseq2.utils.merge import MergeError, merge_object
from fairseq2.utils.structured import StructureError, ValueConverter

T = TypeVar("T", bool, int, float, str)


@final
class AssetCard:
    """Holds information about an asset."""

    _name: str
    _metadata: Mapping[str, object]
    _base: AssetCard | None

    def __init__(
        self, name: str, metadata: Mapping[str, object], base: AssetCard | None = None
    ) -> None:
        self._name = name
        self._metadata = metadata
        self._base = base

    def field(self, name: str) -> AssetCardField:
        return AssetCardField(name, self)

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
    _name: str
    _card: AssetCard

    def __init__(self, name: str, card: AssetCard) -> None:
        self._name = name
        self._card = card

    def exists(self) -> bool:
        card: AssetCard | None = self._card

        while card is not None:
            if self._name in card.metadata:
                return True

            card = card.base

        return False

    def value(self) -> object:
        card: AssetCard | None = self._card

        while card is not None:
            try:
                return card.metadata[self._name]
            except KeyError:
                pass

            card = card.base

        raise AssetCardFieldNotFoundError(self._card.name, self._name)

    def as_(self, kls: type[T]) -> T:
        value = self.value()

        if not isinstance(value, kls):
            raise AssetCardError(
                self._card.name, f"The value of the '{self._name}' field of the '{self._card.name}' asset card is expected to be of type `{kls}`, but is of type `{type(value)}` instead."  # fmt: skip
            )

        return value

    def as_uri(self) -> str:
        value = self.as_(str)

        try:
            parse_result = urlparse(value)
        except ValueError:
            parse_result = None

        if parse_result is not None and parse_result.scheme != "":
            return urlunparse(parse_result)

        try:
            path = Path(value)
        except ValueError:
            raise AssetCardError(
                self._card.name, f"The value of the '{self._name}' field of the '{self._card.name}' asset card is expected to be a URI or a pathname, but is '{value}' instead."  # fmt: skip
            ) from None

        if not path.is_absolute():
            try:
                base_path = self._card._metadata["__base_path__"]
            except KeyError:
                raise AssetCardError(
                    self._card.name, f"The value of the '{self._name}' field of the '{self._card.name}' asset card is a relative pathname ('{path}') and cannot be converted to a URI."  # fmt: skip
                )

            if not isinstance(base_path, Path):
                raise ContractError(
                    f"The value of the '__base_path__' field of the {self._card.name} asset card is expected to be of type `{Path}`, but is of type `{type(base_path)}` instead."
                )

            path = base_path.joinpath(path)

        return path.as_uri()


class AssetCardError(Exception):
    name: str

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)

        self.name = name


class AssetCardFieldNotFoundError(Exception):
    name: str
    field: str

    def __init__(self, name: str, field: str) -> None:
        super().__init__(
            f"The '{name}' asset card does not have a field named '{field}'."
        )

        self.name = name
        self.field = field


@final
class AssetConfigLoader:
    _value_converter: ValueConverter

    def __init__(self, value_converter: ValueConverter) -> None:
        self._value_converter = value_converter

    def load(self, card: AssetCard, base_config: object, config_key: str) -> object:
        config_overrides = []

        base_card: AssetCard | None = card

        while base_card is not None:
            try:
                config_override = base_card.metadata[config_key]
            except KeyError:
                pass
            else:
                config_overrides.append(config_override)

            base_card = base_card.base

        if not config_overrides:
            return base_config

        try:
            unstructured_config = self._value_converter.unstructure(base_config)
        except StructureError as ex:
            raise StructureError(
                "`base_config` cannot be unstructured. See the nested exception for details."
            ) from ex

        config_kls = type(base_config)

        try:
            for config_override in reversed(config_overrides):
                unstructured_config = merge_object(unstructured_config, config_override)
        except MergeError as ex:
            raise AssetCardError(
                card.name, f"The value of the '{config_key}' field of the '{card.name}' asset card cannot be merged with `base_config`. See the nested exception for details."  # fmt: skip
            ) from ex

        try:
            return self._value_converter.structure(unstructured_config, config_kls)
        except StructureError as ex:
            raise AssetCardError(
                card.name, f"The value of the '{config_key}' field of the '{card.name}' asset card cannot be structured as of type `{config_kls}`. See the nested exception for details."  # fmt: skip
            ) from ex
