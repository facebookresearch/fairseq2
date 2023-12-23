# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from os import PathLike
from pathlib import Path
from typing import (
    AbstractSet,
    Any,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Type,
    TypeVar,
    cast,
)
from urllib.parse import urlparse

from typing_extensions import Self

from fairseq2.assets.error import AssetError
from fairseq2.assets.utils import _starts_with_scheme

T = TypeVar("T")


class AssetCard:
    """Holds information about an asset."""

    name: str
    metadata: MutableMapping[str, Any]
    base: Optional[AssetCard]

    def __init__(
        self,
        metadata: MutableMapping[str, Any],
        base: Optional[AssetCard] = None,
    ) -> None:
        """
        :param metadata:
            The metadata to be held in the card. Each key-value item should
            contain a specific piece of information about the asset.
        :param base:
            The card that this card derives from.
        """
        try:
            name = metadata["name"]
        except KeyError:
            raise AssetCardError("`metadata` must contain a key named 'name'.")

        if not isinstance(name, str):
            raise AssetCardError(
                f"The value of 'name' in `metadata` must be of type `{str}`, but is of type `{type(name)}` instead."
            )

        self.name = name
        self.metadata = metadata
        self.base = base

    def field(self, name: str) -> AssetCardField:
        """Return a field of this card.

        If the card does not contain the specified field, its base card will be
        checked recursively.

        :param name:
            The name of the field.
        """
        return AssetCardField(self, [name])

    def _get_field_value(self, name: str, path: List[str]) -> Any:
        assert len(path) > 0

        metadata = self.metadata

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
            if self.base is not None:
                return self.base._get_field_value(name, path)

            pathname = ".".join(path)

            raise AssetCardFieldNotFoundError(
                f"The asset card '{name}' must have a field named '{pathname}'."
            )

        return metadata

    def _set_field_value(self, path: List[str], value: Any) -> None:
        assert len(path) > 0

        metadata = self.metadata

        for depth, field in enumerate(path[:-1]):
            try:
                metadata = metadata[field]
            except KeyError:
                tmp: Dict[str, Any] = {}

                metadata[field] = tmp

                metadata = tmp

            if not isinstance(metadata, Mapping):
                conflict_pathname = ".".join(path[: depth + 1])

                pathname = ".".join(path)

                raise AssetCardError(
                    f"The asset card '{self.name}' cannot have a field named '{pathname}' due to path conflict at '{conflict_pathname}'."
                )

        metadata[path[-1]] = value

    def __repr__(self) -> str:
        return repr(self.metadata)


class AssetCardField:
    """Represents a field of an asset card."""

    card: AssetCard
    path: List[str]

    def __init__(self, card: AssetCard, path: List[str]) -> None:
        """
        :param card:
            The card owning this field.
        :param path:
            The path to this field in the card.
        """
        self.card = card
        self.path = path

    def field(self, name: str) -> AssetCardField:
        """Return a sub-field of this field.

        :param name:
            The name of the sub-field.
        """
        return AssetCardField(self.card, self.path + [name])

    def is_none(self) -> bool:
        """Return ``True`` if the value of the field is ``None``."""
        value = self.card._get_field_value(self.card.name, self.path)

        return value is None

    def as_(self, kls: Type[T], allow_empty: bool = False) -> T:
        """Return the value of this field.

        :param kls:
            The type of the field.
        :param allow_empty:
            If ``True``, allows the field to be empty.
        """
        value = self.card._get_field_value(self.card.name, self.path)
        if value is None:
            pathname = ".".join(self.path)

            raise AssetCardError(
                f"The value of the field '{pathname}' of the asset card '{self.card.name}' must not be `None`."
            )

        if not isinstance(value, kls):
            pathname = ".".join(self.path)

            raise AssetCardError(
                f"The value of the field '{pathname}' of the asset card '{self.card.name}' must be of type `{kls}`, but is of type `{type(value)}` instead."
            )

        if not allow_empty and not value:
            pathname = ".".join(self.path)

            raise AssetCardError(
                f"The value of the field '{pathname}' of the asset card '{self.card.name}' must not be empty."
            )

        return value

    def as_list(self, kls: Type[T], allow_empty: bool = False) -> List[T]:
        """Return the value of this field as a :class:`list` of type ``kls``.

        :param kls:
            The type of the field elements.
        :param allow_empty:
            If ``True``, allows the list to be empty.
        """
        value = self.as_(list, allow_empty)

        for idx, element in enumerate(value):
            if not isinstance(element, kls):
                pathname = ".".join(self.path)

                raise AssetCardError(
                    f"The elements of the field '{pathname}' of the asset card '{self.card.name}' must be of type `{kls}`, but the element at index {idx} is of type `{type(element)}` instead."
                )

        return value

    def as_dict(self, kls: Type[T], allow_empty: bool = False) -> Dict[str, T]:
        """Return the value of this field as a :class:`dict` of type ``kls``.

        :param kls:
            The type of the field values.
        :param allow_empty:
            If ``True``, allows the dictionary to be empty.
        """
        value = self.as_(dict, allow_empty)

        for key, val in value.items():
            if not isinstance(val, kls):
                pathname = ".".join(self.path)

                raise AssetCardError(
                    f"The items of the field '{pathname}' of the asset card '{self.card.name}' must be of type `{kls}`, but the item '{key}' is of type `{type(val)}` instead."
                )

        return value

    def as_set(self, kls: Type[T], allow_empty: bool = False) -> Set[T]:
        """Return the value of this field as a :class:`set` of type ``kls``.

        :param kls:
            The type of the field elements.
        :param allow_empty:
            If ``True``, allows the list to be empty.
        """
        value = self.as_list(kls, allow_empty)

        return set(value)

    def as_one_of(self, valid_values: AbstractSet[T]) -> T:
        """Return the value of this field as one of the values in ``valid_values``

        :param values:
            The values to check against.
        """
        value = self.as_(object)

        if value not in valid_values:
            pathname = ".".join(self.path)

            values = list(valid_values)

            values.sort()

            raise AssetCardError(
                f"The value of the field '{pathname}' of the asset card '{self.card.name}' must be one of {repr(values)}, but is {repr(value)} instead."
            )

        return cast(T, value)

    def as_uri(self) -> str:
        """Return the value of this field as a URI."""
        value = self.as_(object)

        if not isinstance(value, (str, PathLike)):
            pathname = ".".join(self.path)

            raise AssetCardError(
                f"The value of the field '{pathname}' of the asset card '{self.card.name}' must be of type `{str}` or `{PathLike}`, but is of type `{type(value)}` instead."
            )

        try:
            if isinstance(value, PathLike) or not _starts_with_scheme(value):
                return Path(value).as_uri()
            else:
                return urlparse(value).geturl()
        except ValueError as ex:
            pathname = ".".join(self.path)

            raise AssetCardError(
                f"The value of the field '{pathname}' of the asset card '{self.card.name}' must be a URI or an absolute pathname, but is '{value}' instead."
            ) from ex

    def as_filename(self) -> str:
        """Return the value of this field as a filename."""
        value = self.as_(str)

        if os.sep in value or (os.altsep and os.altsep in value):
            pathname = ".".join(self.path)

            raise AssetCardError(
                f"The value of the field '{pathname}' of the asset card '{self.card.name}' must be a filename, but is '{value}' instead."
            )

        return value

    def set(self, value: Any) -> None:
        """Set the value of this field."""
        self.card._set_field_value(self.path, value)

    def check_equals(self, value: Any) -> Self:
        """Check if the value of this field equals to ``value``."""
        if (v := self.as_(object)) != value:
            pathname = ".".join(self.path)

            raise AssetCardError(
                f"The value of the field '{pathname}' of the asset card '{self.card.name}' must be {repr(value)}, but is {repr(v)} instead."
            )

        return self


class AssetCardError(AssetError):
    """Raised when an asset card operation fails."""


class AssetCardFieldNotFoundError(AssetCardError):
    """Raised when an asset card field cannot be found."""
