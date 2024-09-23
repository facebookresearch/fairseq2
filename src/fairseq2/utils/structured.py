# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
from types import NoneType
from typing import NoReturn


def is_unstructured(obj: object) -> bool:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not is_unstructured(key):
                return False

            if not is_unstructured(value):
                return False

        return True

    if isinstance(obj, list):
        for elem in obj:
            if not is_unstructured(elem):
                return False

        return True

    return isinstance(obj, NoneType | bool | int | float | str)


def merge_unstructured(target: object, source: object) -> object:
    def raise_type_error(param_name: str) -> NoReturn:
        raise StructuredError(
            f"`{param_name}` must be a composition of types `bool`, `int`, `float`, `str`, `list`, and `dict`."
        )

    if not is_unstructured(target):
        raise_type_error("target")

    if not is_unstructured(source):
        raise_type_error("source")

    return _do_merge_unstructured(target, source, "")


def _do_merge_unstructured(target: object, source: object, path: str) -> object:
    def raise_dict_error() -> NoReturn:
        target_kls = type(target).__name__
        source_kls = type(source).__name__

        if not path:
            raise StructuredError(
                f"`target` is of type `{target_kls}`, but `source` is of type `{source_kls}`."
            )

        raise StructuredError(
            f"'{path}' is of type `{target_kls}` in `target`, but is of type `{source_kls}` in `source`."
        )

    if isinstance(target, dict):
        if not isinstance(source, dict):
            raise_dict_error()

        sep = "." if path else ""

        output = {}

        ignored_keys = set()

        del_keys = source.get("_del_")
        if del_keys is not None:
            if not isinstance(del_keys, list):
                raise StructuredError(
                    f"'{path}{sep}_del_' in `source` must be of type `list`, but is of type `{type(del_keys).__name__}` instead."
                )

            for idx, del_key in enumerate(del_keys):
                if not isinstance(del_key, str):
                    raise StructuredError(
                        f"Each element under '{path}{sep}_del_' in `source` must be of type `str`, but the element at index {idx} is of type `{type(del_key).__name__}` instead."
                    )

                ignored_keys.add(del_key)

        for key, value in target.items():
            if key not in ignored_keys:
                output[key] = deepcopy(value)

        add_keys = source.get("_add_")
        if add_keys is not None:
            if not isinstance(add_keys, dict):
                raise StructuredError(
                    f"'{path}{sep}_add_' in `source` must be of type `dict`, but is of type `{type(add_keys).__name__}` instead."
                )

            for idx, (add_key, value) in enumerate(add_keys.items()):
                if not isinstance(add_key, str):
                    raise StructuredError(
                        f"Each key under '{path}{sep}_add_' in `source` must be of type `str`, but the key at index {idx} is of type `{type(add_key).__name__}` instead."
                    )

                output[add_key] = deepcopy(value)

        for key, source_value in source.items():
            if key == "_del_" or key == "_add_":
                continue

            # Maintains backwards compatibility with older configuration API.
            if key == "_type_":
                continue

            sub_path = key if not path else f"{path}.{key}"

            try:
                target_value = output[key]
            except KeyError:
                raise StructuredError(
                    f"`target` must contain a '{sub_path}' key since it exists in `source`."
                ) from None

            output[key] = _do_merge_unstructured(target_value, source_value, sub_path)

        return output

    if isinstance(source, dict):
        raise_dict_error()

    if isinstance(source, list):
        return deepcopy(source)

    return source


class StructuredError(ValueError):
    """Raised when a structure or unstructure operation fails."""
