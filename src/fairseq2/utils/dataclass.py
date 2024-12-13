# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import MISSING, fields
from typing import Any, TypeVar, cast

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


T = TypeVar("T", bound=DataClass)


def merge_dataclass(target: T, source: T) -> T:
    """Merge ``target`` with the data contained in ``source``."""
    if type(target) is not type(source):
        raise TypeError(
            f"`target` and `source` are expected to be of the same type, but they are of types `{type(target)}` and `{type(source)}` instead."
        )

    return cast(T, _copy_dataclass(target, source))


def _copy_dataclass(target: DataClass, source: DataClass) -> DataClass:
    kls = type(target)

    kwargs = {}

    for field in fields(kls):
        if not field.init:
            continue

        source_value = getattr(source, field.name)
        if source_value is EMPTY:
            value = getattr(target, field.name)
        else:
            if is_dataclass_instance(source_value):
                target_value = getattr(target, field.name)

                if type(target_value) is type(source_value):
                    value = _copy_dataclass(target_value, source_value)
                else:
                    value = _copy_dataclass_with_defaults(source_value)
            else:
                value = source_value

        kwargs[field.name] = value

    return kls(**kwargs)


def _copy_dataclass_with_defaults(obj: DataClass) -> DataClass:
    kls = type(obj)

    kwargs = {}

    for field in fields(kls):
        if not field.init:
            continue

        value = getattr(obj, field.name)
        if value is EMPTY:
            if field.default == MISSING or field.default_factory == MISSING:
                raise ValueError(
                    f"The `{field.name}` field of `{kls}` in `target` must have a default value or factory."
                )

            continue

        if is_dataclass_instance(value):
            value = _copy_dataclass_with_defaults(value)

        kwargs[field.name] = value

    return kls(**kwargs)
