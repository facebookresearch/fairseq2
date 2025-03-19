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

from fairseq2.error import InternalError
from fairseq2.utils.structured import (
    StructureError,
    default_value_converter,
    unstructure,
)


@final
class AssetCard:
    """Holds information about an asset."""

    _name: str
    _metadata: MutableMapping[str, object]
    _base_card: AssetCard | None
    _base_path: Path | None

    def __init__(
        self,
        name: str,
        metadata: MutableMapping[str, object],
        base_card: AssetCard | None = None,
        base_path: Path | None = None,
    ) -> None:
        """
        :param metadata:
            The metadata to be held in the card. Each key-value item should
            contain a specific piece of information about the asset.
        :param base:
            The card that this card derives from.
        """
        self._name = name
        self._metadata = metadata
        self._base_card = base_card
        self._base_path = base_path

    def field(self, name: str) -> AssetCardField:
        """Return a field of this card.

        If the card does not contain the specified field, its base cards will be
        checked recursively.

        :param name:
            The name of the field.
        """
        return AssetCardField(self, path=[name])

    def _get_field_value(self, leaf_card: AssetCard, path: list[str]) -> object:
        if len(path) == 0:
            raise InternalError("`path` has zero length.")

        metadata: object = self._metadata

        contains = True

        for field in path:
            if metadata is None:
                contains = False

                break

            if not isinstance(metadata, MutableMapping):
                pathname = ".".join(path)

                raise AssetCardFieldNotFoundError(
                    leaf_card.name, f"The '{leaf_card.name}' asset card does not have a field named '{pathname}'."  # fmt: skip
                )

            try:
                metadata = metadata[field]
            except KeyError:
                contains = False

                break

        if not contains:
            if self._base_card is not None:
                return self._base_card._get_field_value(leaf_card, path)

            pathname = ".".join(path)

            raise AssetCardFieldNotFoundError(
                leaf_card.name, f"The '{leaf_card.name}' asset card does not have a field named '{pathname}'."  # fmt: skip
            )

        return metadata

    def _set_field_value(self, path: list[str], value: object) -> None:
        if len(path) == 0:
            raise InternalError("`path` has zero length.")

        metadata = self._metadata

        for depth, field in enumerate(path[:-1]):
            value_ = metadata.get(field)
            if value_ is None:
                tmp: dict[str, object] = {}

                metadata[field] = tmp

                value_ = tmp

            if not isinstance(value_, MutableMapping):
                conflict_pathname = ".".join(path[: depth + 1])

                pathname = ".".join(path)

                raise AssetCardError(
                    self._name, f"The '{self._name}' asset card cannot have a field named '{pathname}' due to path conflict at '{conflict_pathname}'."  # fmt: skip
                )

            metadata = value_

        metadata[path[-1]] = value

    def __repr__(self) -> str:
        return repr(self._metadata)

    @property
    def name(self) -> str:
        """The name of the asset."""
        return self._name

    @property
    def metadata(self) -> Mapping[str, object]:
        """The metadata of the asset."""
        return self._metadata

    @property
    def base(self) -> AssetCard | None:
        """The card that this card derives from."""
        return self._base_card


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
            self._card._get_field_value(self._card, self._path)
        except AssetCardFieldNotFoundError:
            return False

        return True

    def as_unstructured(self) -> object:
        """Return the value of this field in unstructured form."""
        return self._card._get_field_value(self._card, self._path)

    def as_(self, type_: object, *, allow_empty: bool = False) -> Any:
        """Return the value of this field.

        :param type_:
            The type expression of the field.
        :param allow_empty:
            If ``True``, allows the field to be empty.
        """
        unstructured_value = self._card._get_field_value(self._card, self._path)

        try:
            value = default_value_converter.structure(unstructured_value, type_)
        except StructureError as ex:
            pathname = ".".join(self._path)

            raise AssetCardError(
                self._card.name, f"The value of the '{pathname}' field of the '{self._card.name}' asset card cannot be parsed as `{type_}`. See the nested exception for details."  # fmt: skip
            ) from ex

        if value is None:
            return value

        if not allow_empty and isinstance(value, Sized) and len(value) == 0:
            pathname = ".".join(self._path)

            raise AssetCardError(
                self._card.name, f"The value of the '{pathname}' field of the '{self._card.name}' asset card is empty."  # fmt: skip
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

            s = ", ".join(values)

            raise AssetCardError(
                self._card.name, f"The value of the '{pathname}' field of the '{self._card.name}' asset card is expected to be one of the following values, but is '{value}' instead: {s}"  # fmt: skip
            )

        return value

    def as_uri(self) -> str:
        """Return the value of this field as a URI."""
        value = cast(str, self.as_(str))

        try:
            if not _starts_with_scheme(value):
                path = Path(value)
                if not path.is_absolute() and self._card._base_path is not None:
                    path = self._card._base_path.joinpath(path)

                return path.as_uri()

            return urlunparse(urlparse(value))
        except ValueError:
            pathname = ".".join(self._path)

            raise AssetCardError(
                self._card.name, f"The value of the '{pathname}' field of the '{self._card.name}' asset card is expected to be a URI or an absolute pathname, but is '{value}' instead."  # fmt: skip
            ) from None

    def as_filename(self) -> str:
        """Return the value of this field as a filename."""
        value = cast(str, self.as_(str))

        if os.sep in value or (os.altsep and os.altsep in value):
            pathname = ".".join(self._path)

            raise AssetCardError(
                self._card.name, f"The value of the '{pathname}' field of the '{self._card.name}' asset card is expected to be a filename, but is '{value}' instead."  # fmt: skip
            )

        return value

    def set(self, value: object) -> None:
        """Set the value of this field."""
        try:
            unstructured_value = unstructure(value)
        except StructureError as ex:
            raise ValueError(
                "`value` must be of a type that can be unstructured. See the nested exception for details."
            ) from ex

        self._card._set_field_value(self._path, unstructured_value)


class AssetCardNotFoundError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(f"An asset card with name '{name}' is not found.")

        self.name = name


class AssetCardError(Exception):
    name: str

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)

        self.name = name


class AssetCardFieldNotFoundError(AssetCardError):
    pass


_SCHEME_REGEX: Final = re.compile("^[a-zA-Z0-9]+://")


def _starts_with_scheme(s: str) -> bool:
    return re.match(_SCHEME_REGEX, s) is not None
