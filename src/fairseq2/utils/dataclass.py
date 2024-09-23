# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import fields
from typing import Any

from typing_extensions import Self

from fairseq2.typing import DataClass, is_dataclass_instance


class _EmptyType:
    def __reduce__(self) -> str:
        return "EMPTY"

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: MutableMapping[Any, Any]) -> Self:
        return self

    def __repr__(self) -> str:
        return "<empty>"


EMPTY = _EmptyType()
"""A sentinel signifying no value for a dataclass field."""


def fill_empty_fields(target: DataClass, source: DataClass) -> None:
    """Fill the empty fields of ``target`` with the data contained in ``source``."""
    if type(target) is not type(source):
        raise TypeError(
            f"`target` and `source` must be of the same type, but they are of types `{type(target)}` and `{type(source)}` instead."
        )

    _fill_empty_fields(target, source)


def _fill_empty_fields(target: DataClass, source: DataClass) -> None:
    has_empty_field = False

    for field in fields(target):
        target_value = getattr(target, field.name)
        source_value = getattr(source, field.name)

        if is_dataclass_instance(target_value):
            if type(target_value) is not type(source_value):
                if _has_empty_field(target_value):
                    has_empty_field = True
            else:
                _fill_empty_fields(target_value, source_value)

                continue

        if target_value is EMPTY:
            setattr(target, field.name, source_value)

    if has_empty_field:
        raise ValueError(
            "`target` must have no empty field after `fill_empty_fields()`, but one or more fields remained empty."
        )


def _has_empty_field(obj: DataClass) -> bool:
    for field in fields(obj):
        value = getattr(obj, field.name)

        if is_dataclass_instance(value):
            if _has_empty_field(value):
                return True
        elif value is EMPTY:
            return True

    return False


def empty_(obj: DataClass) -> DataClass:
    """Set all fields of ``obj`` and its descendant dataclasses to ``EMPTY``."""
    for field in fields(type(obj)):
        if not field.init:
            raise TypeError(
                "`obj` has one or more fields with `init=False` which is not supported by `empty_()`."
            )

        value = getattr(obj, field.name)

        if is_dataclass_instance(value):
            empty_(value)
        else:
            setattr(obj, field.name, EMPTY)

    return obj
