# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import MISSING, fields
from typing import TypeVar, cast

from fairseq2.typing import EMPTY, DataClass, is_dataclass_instance

T = TypeVar("T", bound=DataClass)


def merge_dataclass(target: T, source: T) -> T:
    """Merge ``target`` with the data contained in ``source``."""
    if type(target) is not type(source):
        raise MergeError(
            f"`target` and `source` must be of the same type, but they are of types `{type(target)}` and `{type(source)}` instead."
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

    try:
        return kls(**kwargs)
    except TypeError as ex:
        raise MergeError(
            "The dataclass has one or more `InitVar` pseudo fields and cannot be constructed."
        ) from ex


def _copy_dataclass_with_defaults(obj: DataClass) -> DataClass:
    kls = type(obj)

    kwargs = {}

    for field in fields(kls):
        if not field.init:
            continue

        value = getattr(obj, field.name)
        if value is EMPTY:
            if field.default == MISSING or field.default_factory == MISSING:
                raise MergeError(
                    f"The `{field.name}` field of `{kls}` in `target` must have a default value or factory."
                )

            continue

        if is_dataclass_instance(value):
            value = _copy_dataclass_with_defaults(value)

        kwargs[field.name] = value

    try:
        return kls(**kwargs)
    except TypeError as ex:
        raise MergeError(
            "The dataclass has one or more `InitVar` pseudo fields and cannot be constructed."
        ) from ex


def merge_object(target: object, source: object) -> object:
    if not isinstance(target, Mapping) or not isinstance(source, Mapping):
        return source

    return merge_map(target, source)


def merge_map(
    target: Mapping[str, object], source: Mapping[str, object]
) -> Mapping[str, object]:
    return _do_merge_map(target, source, [])


def _do_merge_map(
    target: object, source: object, path: list[str]
) -> Mapping[str, object]:
    def build_pathname(subpath: str | None = None) -> str:
        if subpath is not None:
            return ".".join(path + [subpath])

        return ".".join(path)

    if not isinstance(source, Mapping):
        pathname = build_pathname()

        raise MergeError(
            f"The '{pathname}' path at `source` must be of type `{Mapping}`, but is of type `{type(source)}` instead."
        )

    if not isinstance(target, Mapping):
        pathname = build_pathname()

        raise MergeError(
            f"The '{pathname}' path at `target` must be of type `{Mapping}`, but is of type `{type(target)}` instead."
        )

    output = {}

    ignored_keys = set()

    del_keys = source.get("_del_")
    if del_keys is not None:
        if not isinstance(del_keys, list):
            pathname = build_pathname("_del_")

            raise MergeError(
                f"'{pathname}' at `source` must be of type `{list}`, but is of type `{type(del_keys)}` instead."
            )

        for idx, del_key in enumerate(del_keys):
            if not isinstance(del_key, str):
                pathname = build_pathname("_del_")

                raise MergeError(
                    f"Each element under '{pathname}' at `source` must be of type `str`, but the element at index {idx} is of type `{type(del_key)}` instead."
                )

            ignored_keys.add(del_key)

    for k, v in target.items():
        if k not in ignored_keys:
            output[k] = deepcopy(v)

    add_keys = source.get("_add_")
    if add_keys is not None:
        if not isinstance(add_keys, Mapping):
            pathname = build_pathname("_add_")

            raise MergeError(
                f"'{pathname}' at `source` must be of type `{Mapping}`, but is of type `{type(add_keys)}` instead."
            )

        for idx, (add_key, value) in enumerate(add_keys.items()):
            if not isinstance(add_key, str):
                pathname = build_pathname("_add_")

                raise MergeError(
                    f"Each key under '{pathname}' at `source` must be of type `str`, but the key at index {idx} is of type `{type(add_key)}` instead."
                )

            if add_key in output:
                pathname = build_pathname(add_key)

                raise MergeError(f"`target` already has an item at path '{pathname}'.")

            output[add_key] = deepcopy(value)

    set_keys = source.get("_set_")
    if set_keys is not None:
        if not isinstance(set_keys, Mapping):
            pathname = build_pathname("_set_")

            raise MergeError(
                f"'{pathname}' at `source` must be of type `{Mapping}`, but is of type `{type(set_keys)}` instead."
            )

        for idx, (set_key, value) in enumerate(set_keys.items()):
            if not isinstance(set_key, str):
                pathname = build_pathname("_set_")

                raise MergeError(
                    f"Each key under '{pathname}' at `source` must be of type `str`, but the key at index {idx} is of type `{type(set_key)}` instead."
                )

            if set_key not in output:
                pathname = build_pathname(set_key)

                raise MergeError(
                    f"`target` does not have an item at path '{pathname}'."
                )

            output[set_key] = deepcopy(value)

    for key, source_value in source.items():
        if key == "_del_" or key == "_add_" or key == "_set_":
            continue

        try:
            target_value = output[key]
        except KeyError:
            pathname = build_pathname(key)

            raise MergeError(
                f"`target` must have an item at path '{pathname}'."
            ) from None

        path.append(key)

        output[key] = _do_merge_map(target_value, source_value, path)

        path.pop()

    return output


class MergeError(Exception):
    pass


def to_mergeable(obj: object) -> object:
    if not isinstance(obj, Mapping):
        return obj

    return to_mergeable_map(obj)


def to_mergeable_map(obj: Mapping[str, object]) -> Mapping[str, object]:
    output = {}

    set_map = None

    for key, value in obj.items():
        if isinstance(value, Mapping) and len(value) > 0:
            output[key] = to_mergeable_map(value)
        else:
            if set_map is None:
                set_map = {}

            set_map[key] = value

    if set_map:
        output["_set_"] = set_map

    return output
