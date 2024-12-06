# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import re
from collections.abc import Mapping, MutableMapping, Set, Sized
from pathlib import Path
from typing import Any, Final, cast, final
from urllib.parse import urlparse, urlunparse

from fairseq2.assets.error import AssetError
from fairseq2.utils.structured import (
    StructuredError,
    ValueConverter,
    default_value_converter,
)


@final
class AssetCard:
    """Holds information about an asset."""

    _name: str
    _metadata: MutableMapping[str, Any]
    _base: AssetCard | None
    _value_converter: ValueConverter

    def __init__(
        self,
        metadata: MutableMapping[str, Any],
        base: AssetCard | None = None,
        value_converter: ValueConverter | None = None,
    ) -> None:
        """
        :param metadata:
            The metadata to be held in the card. Each key-value item should
            contain a specific piece of information about the asset.
        :param base:
            The card that this card derives from.
        :param value_converter:
            The :class:`ValueConverter` instance to use.
        """
        try:
            name = metadata["name"]
        except KeyError:
            raise AssetCardError(
                "`metadata` must contain a key named 'name'."
            ) from None

        if not isinstance(name, str):
            raise AssetCardError(
                f"The value of 'name' in `metadata` must be of type `{str}`, but is of type `{type(name)}` instead."
            )

        self._name = name
        self._metadata = metadata
        self._base = base
        self._value_converter = value_converter or default_value_converter

    def field(self, name: str) -> AssetCardField:
        """Return a field of this card.

        If the card does not contain the specified field, its base cards will be
        checked recursively.

        :param name:
            The name of the field.
        """
        return AssetCardField(self, path=[name])

    def _get_field_value(self, name: str, path: list[str]) -> object:
        assert len(path) > 0

        metadata = self._metadata

        contains = True

        for field in path:
            if metadata is None:
                contains = False

                break

            if not isinstance(metadata, Mapping):
                pathname = ".".join(path)

                raise AssetCardFieldNotFoundError(
                    f"The asset card '{name}' must have a field named '{pathname}'."
                )

            try:
                metadata = metadata[field]
            except KeyError:
                contains = False

                break

        if not contains:
            if self._base is not None:
                return self._base._get_field_value(name, path)

            pathname = ".".join(path)

            raise AssetCardFieldNotFoundError(
                f"The asset card '{name}' must have a field named '{pathname}'."
            )

        return metadata

    def _set_field_value(self, path: list[str], value: object) -> None:
        assert len(path) > 0

        metadata = self._metadata

        for depth, field in enumerate(path[:-1]):
            try:
                metadata = metadata[field]
            except KeyError:
                tmp: dict[str, Any] = {}

                metadata[field] = tmp

                metadata = tmp

            if not isinstance(metadata, Mapping):
                conflict_pathname = ".".join(path[: depth + 1])

                pathname = ".".join(path)

                raise AssetCardError(
                    f"The asset card '{self._name}' cannot have a field named '{pathname}' due to path conflict at '{conflict_pathname}'."
                )

        metadata[path[-1]] = value

    def __repr__(self) -> str:
        return repr(self._metadata)

    @property
    def name(self) -> str:
        """The name of the asset."""
        return self._name

    @property
    def metadata(self) -> Mapping[str, Any]:
        """The metadata of the asset."""
        return self._metadata

    @property
    def base(self) -> AssetCard | None:
        """The card that this card derives from."""
        return self._base


@final
class AssetCardField:
    """Represents a field of an asset card."""

    _card: AssetCard
    _path: list[str]

    def __init__(self, card: AssetCard, path: list[str]) -> None:
        """
        :param card:
            The card owning this field.
        :param path:
            The path to this field in the card.
        """
        self._card = card
        self._path = path

    def field(self, name: str) -> AssetCardField:
        """Return a sub-field of this field.

        :param name:
            The name of the sub-field.
        """
        return AssetCardField(self._card, self._path + [name])

    def exists(self) -> bool:
        """Return ``True`` if the field exists."""
        try:
            self._card._get_field_value(self._card.name, self._path)
        except AssetCardFieldNotFoundError:
            return False

        return True

    def as_unstructured(self) -> object:
        """Return the value of this field in unstructured form."""
        return self._card._get_field_value(self._card.name, self._path)

    def as_(self, type_expr: Any, *, allow_empty: bool = False) -> Any:
        """Return the value of this field.

        :param type_expr:
            The type expression of the field.
        :param allow_empty:
            If ``True``, allows the field to be empty.
        """
        unstructured_value = self._card._get_field_value(self._card.name, self._path)

        try:
            value = self._card._value_converter.structure(unstructured_value, type_expr)
        except StructuredError as ex:
            pathname = ".".join(self._path)

            raise AssetCardError(
                f"The value of the field '{pathname}' of the asset card '{self._card.name}' cannot be parsed as `{type_expr}`. See nested exception for details."
            ) from ex

        if value is None:
            return value

        if not allow_empty and isinstance(value, Sized) and len(value) == 0:
            pathname = ".".join(self._path)

            raise AssetCardError(
                f"The value of the field '{pathname}' of the asset card '{self._card.name}' must not be empty."
            )

        return value

    def as_one_of(self, valid_values: Set[str]) -> str:
        """Return the value of this field as one of the values in ``valid_values``

        :param values:
            The values to check against.
        """
        if not valid_values:
            raise ValueError("`valid_values` must not be empty.")

        value = cast(str, self.as_(str))

        if value not in valid_values:
            pathname = ".".join(self._path)

            values = list(valid_values)

            values.sort()

            raise AssetCardError(
                f"The value of the field '{pathname}' of the asset card '{self._card.name}' must be one of {repr(values)}, but is {repr(value)} instead."
            )

        return value

    def as_uri(self) -> str:
        """Return the value of this field as a URI."""
        value = cast(str, self.as_(str))

        try:
            if not _starts_with_scheme(value):
                path = Path(value)
                if not path.is_absolute():
                    base_path = self._card.metadata.get("__base_path__")
                    if base_path is not None:
                        path = base_path.joinpath(path)

                return path.as_uri()

            return urlunparse(urlparse(value))
        except ValueError as ex:
            pathname = ".".join(self._path)

            raise AssetCardError(
                f"The value of the field '{pathname}' of the asset card '{self._card.name}' must be a URI or an absolute pathname, but is '{value}' instead."
            ) from ex

    def as_filename(self) -> str:
        """Return the value of this field as a filename."""
        value = cast(str, self.as_(str))

        if os.sep in value or (os.altsep and os.altsep in value):
            pathname = ".".join(self._path)

            raise AssetCardError(
                f"The value of the field '{pathname}' of the asset card '{self._card.name}' must be a filename, but is '{value}' instead."
            )

        return value

    def get_as_(
        self, type_expr: Any, default: object = None, *, allow_empty: bool = False
    ) -> Any:
        """Return the value of this field if it exists; otherwise, return ``default``.

        :param type_expr:
            The type expression of the field.
        :param default:
            The default value.
        :param allow_empty:
            If ``True``, allows the field to be empty.
        """
        try:
            return self.as_(type_expr, allow_empty=allow_empty)
        except AssetCardFieldNotFoundError:
            return default

    def set(self, value: object) -> None:
        """Set the value of this field.

        :param value:
            The value to set.
        """
        try:
            unstructured_value = self._card._value_converter.unstructure(value)
        except ValueError as ex:
            raise ValueError(
                "`type_expr` must be a supported type expression. See nested exception for details."
            ) from ex
        except TypeError as ex:
            raise TypeError(
                "`value` cannot be unstructured. See nested exception for details."
            ) from ex

        self._card._set_field_value(self._path, unstructured_value)


class AssetCardError(AssetError):
    """Raised when an asset card operation fails."""


class AssetCardFieldNotFoundError(AssetCardError):
    """Raised when an asset card field cannot be found."""


_SCHEME_REGEX: Final = re.compile("^[a-zA-Z0-9]+://")


def _starts_with_scheme(s: str) -> bool:
    return re.match(_SCHEME_REGEX, s) is not None
