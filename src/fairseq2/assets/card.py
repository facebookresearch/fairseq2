# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import (
    AbstractSet,
    Any,
    List,
    Mapping,
    NoReturn,
    Optional,
    Type,
    TypeVar,
    cast,
    get_origin,
)
from urllib.parse import urlparse

from fairseq2.assets.error import AssetError

T = TypeVar("T")


class AssetCard:
    """Provides information about an asset.

    Since asset cards have no formal schema, they can describe any type of asset
    including pretrained models and datasets. They typically contain information
    such as the configuration, location, intended use, and evaluation results of
    an asset.
    """

    name: str
    data: Any
    base: Optional["AssetCard"]

    def __init__(
        self, name: str, data: Mapping[str, Any], base: Optional["AssetCard"] = None
    ) -> None:
        """
        :param name:
            The name of the asset.
        :param data:
            The data of the asset card. Each key-value entry (i.e. field) should
            hold a specific piece of information about the asset.
        :param base:
            The card that this card derives from.
        """
        self.name = name
        self.data = data
        self.base = base

    def field(self, name: str) -> "AssetCardField":
        """Return a field of the card.

        If this card does not contain the specified field, its base card will be
        recursively checked.

        :param name:
            The name of the field.

        :returns:
            An :class:`AssetCardField` representing the field.
        """
        data = None

        card: Optional[AssetCard] = self

        while card and data is None:
            try:
                data = card.data[name]
            except KeyError:
                pass

            card = card.base

        return AssetCardField(self, [name], data)

    def __str__(self) -> str:
        return str(self.data)


class AssetCardField:
    """Represents a field in an asset card."""

    card: AssetCard
    path: List[str]
    data: Any

    def __init__(self, card: AssetCard, path: List[str], data: Any) -> None:
        """
        :param card:
            The asset card owning this field.
        :param path:
            The list containing the names of this field and all its parent
            fields in top-down order.
        :param data:
            The data held by this field.
        """
        self.card = card
        self.path = path
        self.data = data

    def as_(self, kls: Type[T], allow_empty: bool = False) -> T:
        """Return the value of this field if it exists and is of type ``kls``;
        otherwise, raise an :class:`AssertCardError`.

        :param kls:
            The type to check against.
        :param allow_empty:
            If ``True``, allows the value to be empty.
        """
        if self.data is None:
            pathname = ".".join(self.path)

            raise AssetCardFieldNotFoundError(
                f"The asset card '{self.card.name}' must have a field named '{pathname}'."
            )

        if not isinstance(self.data, get_origin(kls) or kls):
            self._raise_card_error(
                f"The type of the {{display_name}} must be {kls}, but is {type(self.data)} instead."
            )

        if not allow_empty and not self.data:
            self._raise_card_error("The value of the {display_name} must be non-empty.")

        return cast(T, self.data)

    def as_uri(self) -> str:
        """Return the value of this field if it represents a valid URI;
        otherwise, raise an :class:`AssertCardError`."""
        value = self.as_(str)

        try:
            uri = urlparse(value)
        except ValueError:
            uri = None

        if uri and uri.scheme and uri.netloc:
            return value

        self._raise_card_error(
            f"The value of the {{display_name}} must be a valid URI, but is '{value}' instead."
        )

    def as_filename(self) -> str:
        """Return the value of this field if it represents a valid filename;
        otherwise, raise an :class:`AssertCardError`."""
        value = self.as_(str)

        if os.sep in value or (os.altsep and os.altsep in value):
            self._raise_card_error(
                f"The value of the {{display_name}} must be a valid filename, but is '{value}' instead."
            )

        return value

    def as_list(self, kls: Type[T], allow_empty: bool = False) -> List[T]:
        """Return the value of this field as a :class:`list` if all its elements
        are of type ``kls``; otherwise, raise an :class:`AssertCardError`.

        :param kls:
            The type to check against.
        :param allow_empty:
            If ``True``, allows the list to be empty.
        """
        value = self.as_(list, allow_empty)

        for elem in value:
            if not isinstance(elem, kls):
                self._raise_card_error(
                    f"The value of the {{display_name}} must be a list of type of {kls}, but has at least one element of type {type(elem)}."
                )

        return value

    def as_one_of(self, valid_values: AbstractSet[T]) -> T:
        """Return the value of this field if it is included in ``valid_values``;
        otherwise, raise an :class:`AssertCardError`.

        :param values:
            The values to check against.
        """
        if self.data in valid_values:
            return cast(T, self.data)

        values = ", ".join(sorted([repr(v) for v in valid_values]))

        self._raise_card_error(
            f"The value of the {{display_name}} must be one of [{values}], but is {repr(self.data)} instead."
        )

    def check_equals(self, value: Any) -> "AssetCardField":
        """Check if the value of this field equals ``value``; if not, raise an
        :class:`AssertCardError`.

        :param value:
            The value to check against.
        """
        if self.data == value:
            return self

        self._raise_card_error(
            f"The value of the {{display_name}} must be {repr(value)}, but is {repr(self.data)} instead."
        )

    def field(self, name: str) -> "AssetCardField":
        """Return a sub-field of this field.

        :param name:
            The name of the sub-field.

        :returns:
            An :class:`AssetCardField` representing the sub-field.
        """
        data = None

        if self.data is not None and isinstance(self.data, Mapping):
            try:
                data = self.data[name]
            except KeyError:
                pass

        return AssetCardField(self.card, self.path + [name], data)

    def _raise_card_error(self, msg: str) -> NoReturn:
        pathname = ".".join(self.path)

        display_name = f"field '{pathname}' of the asset card '{self.card.name}'"

        raise AssetCardError(msg.format(display_name=display_name))


class AssetCardError(AssetError):
    """Raised when an asset card cannot be processed."""


class AssetCardFieldNotFoundError(AssetCardError):
    """Raised when an asset card field cannot be found."""
