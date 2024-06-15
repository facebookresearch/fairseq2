# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any, Dict, List, Mapping, Optional, TextIO, cast, get_type_hints

import yaml

from fairseq2.typing import DataClass, is_dataclass_instance
from fairseq2.utils.value_converter import ValueConverter, default_value_converter


def update_dataclass(
    obj: DataClass,
    overrides: Mapping[str, Any],
    *,
    value_converter: Optional[ValueConverter] = None,
) -> List[str]:
    """Update ``obj`` with the data contained in ``overrides``.

    :param obj:
        The data class instance to update.
    :param overrides:
        The dictionary containing the data to set in ``obj``.
    :param value_converter:
        The :class:`ValueConverter` instance to use. If ``None``, the default
        instance will be used.
    """
    if value_converter is None:
        value_converter = default_value_converter

    unknown_fields: List[str] = []

    field_path: List[str] = []

    def update(obj_: DataClass, overrides_: Mapping[str, Any]) -> None:
        overrides_copy = {**overrides_}

        type_hints = get_type_hints(type(obj_))

        for field in fields(obj_):
            value = getattr(obj_, field.name)

            try:
                override = overrides_copy.pop(field.name)
            except KeyError:
                continue

            # Recursively traverse child dataclasses.
            if override is not None and is_dataclass_instance(value):
                if not isinstance(override, Mapping):
                    pathname = ".".join(field_path + [field.name])

                    raise FieldError(
                        pathname, f"The field '{pathname}' is expected to be of type `{type(value)}`, but is of type `{type(override)}` instead."  # fmt: skip
                    )

                field_path.append(field.name)

                update(value, override)

                field_path.pop()
            else:
                type_hint = type_hints[field.name]

                try:
                    override = value_converter.structure(override, type_hint)
                except (TypeError, ValueError) as ex:
                    pathname = ".".join(field_path + [field.name])

                    raise FieldError(
                        pathname, f"The value of the field '{pathname}' cannot be parsed. See nested exception for details"  # fmt: skip
                    ) from ex

                setattr(obj_, field.name, override)

        if overrides_copy:
            unknown_fields.extend(
                ".".join(field_path + [name]) for name in overrides_copy
            )

    update(obj, overrides)

    unknown_fields.sort()

    return unknown_fields


class FieldError(RuntimeError):
    """Raised when a dataclass field cannot be parsed."""

    _field_name: str

    def __init__(self, field_name: str, message: str) -> None:
        super().__init__(message)

        self._field_name = field_name

    @property
    def field_name(self) -> str:
        return self._field_name


def dump_dataclass(obj: DataClass, fp: TextIO) -> None:
    """Dump ``obj`` to ``fp`` in YAML format."""
    yaml.safe_dump(to_safe_dict(obj), fp, sort_keys=False)


def to_safe_dict(
    obj: DataClass, value_converter: Optional[ValueConverter] = None
) -> Dict[str, Any]:
    """Convert ``obj`` to a :class:`dict` safe to serialize in YAML."""
    if value_converter is None:
        value_converter = default_value_converter

    try:
        data = value_converter.unstructure(asdict(obj))
    except TypeError as ex:
        raise ValueError(
            "`obj` must contain only values that can be serialized to standard YAML. See nested exception for details."
        ) from ex

    def sanity_check(data_: Any) -> None:
        if data_ is None:
            return

        if isinstance(data_, (bool, int, float, str)):
            return

        if isinstance(data_, list):
            for e in data_:
                sanity_check(e)

            return

        if isinstance(data_, dict):
            for k, v in data_.items():
                sanity_check(k)
                sanity_check(v)

            return

        raise RuntimeError(
            f"Unstructured output of `obj` must contain only primitive types, lists, and dicts, but it contains a value of type `{type(data_)}`."
        )

    sanity_check(data)

    return cast(Dict[str, Any], data)
