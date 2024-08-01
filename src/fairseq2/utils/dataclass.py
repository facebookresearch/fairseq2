# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import fields
from typing import Any, MutableMapping

from typing_extensions import Self

from fairseq2.typing import DataClass, is_dataclass_instance


class _EmptyType:
    def __reduce__(self) -> str:
        return _EmptyType.__name__

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: MutableMapping[Any, Any]) -> Self:
        return self

    def __repr__(self) -> str:
        return "<empty>"


EMPTY = _EmptyType()
"""A sentinel signifying no value for a dataclass field."""


def update_dataclass(target: DataClass, source: DataClass) -> None:
    """Update ``target`` with the data contained in ``source``."""
    if type(target) is not type(source):
        raise TypeError(
            f"`target` and `source` must be of the same type, but they are of types `{type(target)}` and `{type(source)}` instead."
        )

    _update_dataclass(target, source)


def _update_dataclass(target: DataClass, source: DataClass) -> None:
    for field in fields(target):
        source_value = getattr(source, field.name)
        if source_value is EMPTY:
            continue

        if is_dataclass_instance(source_value):
            target_value = getattr(target, field.name)

            if type(target_value) is type(source_value):
                _update_dataclass(target_value, source_value)

                continue

            source_value = _copy_dataclass_with_defaults(source_value)

        setattr(target, field.name, source_value)


def _copy_dataclass_with_defaults(obj: DataClass) -> DataClass:
    kls = type(obj)

    kwargs = {}

    for field in fields(kls):
        value = getattr(obj, field.name)
        if value is EMPTY:
            continue

        if is_dataclass_instance(value):
            value = _copy_dataclass_with_defaults(value)

        kwargs[field.name] = value

    return kls(**kwargs)


def empty(obj: DataClass) -> None:
    """Set all fields of ``obj`` and its descendants to ``EMPTY``."""
    for field in fields(type(obj)):
        if not field.init:
            raise TypeError(
                "`obj` has one or more fields with `init=False` which is not supported by `empty()`."
            )

        value = getattr(obj, field.name)

        if is_dataclass_instance(value):
            empty(value)
        else:
            setattr(obj, field.name, EMPTY)
