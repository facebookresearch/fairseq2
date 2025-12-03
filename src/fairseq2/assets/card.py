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

from fairseq2.utils.config import ConfigMerger, ConfigProcessor
from fairseq2.utils.structured import ValueConverter
from fairseq2.utils.uri import Uri, UriFormatError

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
        """
        :raises AssetCardFieldNotFoundError:
        """
        field = self.maybe_get_field(name)
        if field is None:
            raise AssetCardFieldNotFoundError(self, name)

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


class AssetCardFieldError(Exception):
    def __init__(self, card: AssetCard, field: str, message: str) -> None:
        super().__init__(message)

        self.card = card
        self.field = field


class AssetCardFieldNotFoundError(AssetCardFieldError):
    def __init__(self, card: AssetCard, field: str) -> None:
        super().__init__(
            card, field, f"field '{field}' of asset card '{card.name}' is not found"
        )


class AssetCardFieldTypeError(AssetCardFieldError):
    def __init__(
        self, card: AssetCard, field: str, kls: type[object], valid_kls: type[object]
    ) -> None:
        super().__init__(
            card, field, f"field '{field}' of asset card '{card.name}' is expected to be of type `{valid_kls}`, but is of type `{kls}` instead"  # fmt: skip
        )

        self.kls = kls
        self.valid_kls = valid_kls


class AssetCardFieldFormatError(AssetCardFieldError):
    pass


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
        """
        :raises AssetCardFieldTypeError:
        """
        if not isinstance(self._value, kls):
            raise AssetCardFieldTypeError(
                self._card, self._name, type(self._value), kls
            )

        return self._value

    def as_uri(self) -> Uri:
        """
        :raises AssetCardFieldTypeError:
        :raises AssetCardFieldFormatError:
        """
        value = self.as_(str)

        uri = Uri.maybe_parse(value)
        if uri is not None:
            return uri

        try:
            path = Path(value)
        except ValueError:
            raise AssetCardFieldFormatError(
                self._card, self._name, f"field '{self._name}' of asset card '{self._card.name}' is expected to be a pathname or a URI, but is '{value}' instead"  # fmt: skip
            ) from None

        if not path.is_absolute():
            base_path = self._card.metadata.get("__base_path__")
            if not isinstance(base_path, Path):
                raise AssetCardFieldFormatError(
                    self._card, self._name, f"field '{self._name}' of asset card '{self._card.name}' is expected to be an absolute pathname, but is '{value}' instead"  # fmt: skip
                )

            path = base_path.joinpath(path)

        try:
            return Uri.from_path(path)
        except UriFormatError as ex:
            raise AssetCardFieldFormatError(
                self._card, self._name, f"failed to convert value '{path}' of field '{self._name}' of asset card '{self._card.name}' to a URI"  # fmt: skip
            ) from ex


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

        unstructured_config = self._value_converter.unstructure(base_config)

        for name, config_overrides in reversed(all_config_overrides):
            # TODO(balioglu): unescape _set_ and _del_ in config_overrides
            unstructured_config = self._config_merger.merge(
                unstructured_config, config_overrides
            )

            # TODO(balioglu): unescape config directives and run them.
            unstructured_config = self._config_processor.process(unstructured_config)

        config_kls = type(base_config)

        return self._value_converter.structure(unstructured_config, config_kls)
