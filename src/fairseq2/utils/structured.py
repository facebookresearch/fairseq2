# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from copy import deepcopy
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import NoneType, UnionType
from typing import (
    Any,
    Literal,
    NoReturn,
    Protocol,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import torch

from fairseq2.dependency import DependencyContainer, resolve
from fairseq2.typing import DataClass, DataType, Device
from fairseq2.utils.dataclass import EMPTY


class _StructureFn(Protocol):
    def __call__(
        self, type_: Any, type_args: tuple[Any, ...], obj: object, set_empty: bool
    ) -> object:
        ...


class _UnstructureFn(Protocol):
    def __call__(self, obj: Any) -> object:
        ...


class ValueConverter:
    """Structures objects using provided type expressions."""

    _structure_fns: dict[Any, _StructureFn]
    _unstructure_fns: dict[Any, _UnstructureFn]

    def __init__(self) -> None:
        self._structure_fns = {
            # fmt: off
            bool:      self._structure_identity,
            DataClass: self._structure_dataclass,
            DataType:  self._structure_dtype,
            Device:    self._structure_device,
            dict:      self._structure_dict,
            float:     self._structure_primitive,
            Enum:      self._structure_enum,
            int:       self._structure_primitive,
            list:      self._structure_list,
            Literal:   self._structure_literal,
            Mapping:   self._structure_dict,
            NoneType:  self._structure_identity,
            Path:      self._structure_path,
            Sequence:  self._structure_list,
            set:       self._structure_set,
            Set:       self._structure_set,
            str:       self._structure_identity,
            tuple:     self._structure_tuple,
            Union:     self._structure_union,
            UnionType: self._structure_union,
            # fmt: on
        }

        self._unstructure_fns = {
            # fmt: off
            bool:      self._unstructure_identity,
            DataClass: self._unstructure_dataclass,
            DataType:  self._unstructure_dtype,
            Device:    self._unstructure_device,
            float:     self._unstructure_identity,
            Enum:      self._unstructure_enum,
            int:       self._unstructure_identity,
            list:      self._unstructure_sequence,
            Mapping:   self._unstructure_mapping,
            NoneType:  self._unstructure_identity,
            Path:      self._unstructure_path,
            Set:       self._unstructure_set,
            str:       self._unstructure_identity,
            tuple:     self._unstructure_sequence,
            # fmt: on
        }

    def structure(self, obj: object, type_expr: Any, *, set_empty: bool = False) -> Any:
        type_, type_args = get_origin(type_expr), get_args(type_expr)

        if type_ is None:
            type_ = type_expr

        if type_ is Any:
            return obj

        lookup_type = type_

        if isinstance(type_, type):
            if is_dataclass(type_):
                lookup_type = DataClass
            elif issubclass(type_, Enum):
                lookup_type = Enum

        try:
            fn = self._structure_fns[lookup_type]
        except KeyError:
            supported_types = ", ".join(str(t) for t in self._structure_fns.keys())

            raise StructuredError(
                f"`type_expr` must be an expression consisting of the following types, but is `{type_expr}` instead: {supported_types}"
            ) from None

        try:
            return fn(type_, type_args, obj, set_empty)
        except StructuredError as ex:
            raise StructuredError(
                f"`obj` cannot be structured to `{type_expr}`. See nested exception for details."
            ) from ex

    @staticmethod
    def _structure_primitive(
        type_: Any, type_args: tuple[Any, ...], obj: object, set_empty: bool
    ) -> object:
        if isinstance(obj, type_):
            return obj

        try:
            return type_(obj)
        except (TypeError, ValueError) as ex:
            raise StructuredError(
                f"`obj` cannot be parsed as `{type_}`. See nested exception for details."
            ) from ex

    @staticmethod
    def _structure_identity(
        type_: Any, type_args: tuple[Any, ...], obj: object, set_empty: bool
    ) -> object:
        if isinstance(obj, type_):
            return obj

        raise StructuredError(
            f"`obj` must be of type `{type_}`, but is of type `{type(obj)}` instead."
        )

    def _structure_dataclass(
        self, type_: Any, type_args: tuple[Any, ...], obj: object, set_empty: bool
    ) -> object:
        if type_ is DataClass:
            raise StructuredError(
                f"`type_expr` must be a concrete dataclass type, but is `{DataClass}` instead."
            )

        if isinstance(obj, type_):
            values = {f.name: getattr(obj, f.name) for f in fields(type_)}

            return self._create_dataclass(type_, values, set_empty)

        if isinstance(obj, Mapping):
            values = self.structure(obj, dict[str, Any])

            return self._create_dataclass(type_, values, set_empty)

        raise StructuredError(
            f"`obj` must be of type `{type_}` or `{Mapping}`, but is of type `{type(obj)}` instead."
        )

    def _create_dataclass(
        self, type_: type[DataClass], values: dict[str, Any], set_empty: bool
    ) -> object:
        type_hints = get_type_hints(type_)

        kwargs = {}

        for field in fields(type_):
            try:
                value = values.pop(field.name)
            except KeyError:
                value = EMPTY

            # Fields with `init=False` are initialized in `__post_init__()`.
            if not field.init:
                continue

            if value is EMPTY:
                if not set_empty:
                    if field.default == MISSING and field.default_factory == MISSING:
                        raise StructuredError(
                            f"The `{field.name}` field of the dataclass has no default value or factory."
                        )

                    continue

                if hasattr(type_, "__post_init__"):
                    raise StructuredError(
                        f"The `{field.name}` field of the dataclass must not be `EMPTY` since `{type_}` has a `__post_init__()` method."
                    )
            else:
                try:
                    value = self.structure(
                        value, type_hints[field.name], set_empty=set_empty
                    )
                except StructuredError as ex:
                    raise StructuredError(
                        f"The `{field.name}` field of the dataclass cannot be structured. See nested exception for details."
                    ) from ex

            kwargs[field.name] = value

        if values:
            extra_keys = ", ".join(list(values.keys()))

            raise StructuredError(
                f"`obj` must contain only keys corresponding to the fields of `{type_}`, but it contains the following extra keys: {extra_keys}"
            )

        return type_(**kwargs)

    def _structure_dict(
        self, type_: Any, type_args: tuple[Any, ...], obj: object, set_empty: bool
    ) -> dict[object, object]:
        if isinstance(obj, Mapping):
            if len(type_args) != 2:
                raise StructuredError(
                    f"`type_expr` must have a key-value type expression for `{type_}`."
                )

            output = {}

            for k, v in obj.items():
                k = self.structure(k, type_args[0])
                v = self.structure(v, type_args[1])

                output[k] = v

            return output

        raise StructuredError(
            f"`obj` must be of type `{Mapping}`, but is of type `{type(obj)}` instead."
        )

    @staticmethod
    def _structure_dtype(
        type_: Any, type_args: tuple[Any, ...], obj: object, set_empty: bool
    ) -> DataType:
        if isinstance(obj, DataType):
            return obj

        if isinstance(obj, str):
            if obj.startswith("torch."):
                obj = obj[6:]

            if isinstance(dtype := getattr(torch, obj, None), DataType):
                return dtype

            raise StructuredError(
                f"`obj` must be a `torch.dtype` identifier, but is '{obj}' instead."
            )

        raise StructuredError(
            f"`obj` must be of type `{DataType}` or `{str}`, but is of type `{type(obj)}` instead."
        )

    @staticmethod
    def _structure_device(
        type_: Any, type_args: tuple[Any, ...], obj: object, set_empty: bool
    ) -> Device:
        if isinstance(obj, Device):
            return obj

        if isinstance(obj, str):
            try:
                return Device(obj)
            except RuntimeError as ex:
                raise StructuredError(str(ex))

        raise StructuredError(
            f"`obj` must be of type `{Device}` or `{str}`, but is of type `{type(obj)}` instead."
        )

    @staticmethod
    def _structure_enum(
        type_: Any, type_args: tuple[Any, ...], obj: object, set_empty: bool
    ) -> object:
        if isinstance(obj, type_):
            return obj

        if isinstance(obj, str):
            try:
                return type_[obj]
            except KeyError:
                values = ", ".join(e.name for e in type_)

                raise StructuredError(
                    f"`obj` must be one of the following enumeration values, but is '{obj}' instead: {values}"
                ) from None

        raise StructuredError(
            f"`obj` must be of type `{type_}` or `{str}`, but is of type `{type(obj)}` instead."
        )

    def _structure_list(
        self, type_: Any, type_args: tuple[Any, ...], obj: object, set_empty: bool
    ) -> list[object]:
        if isinstance(obj, Sequence):
            if len(type_args) != 1:
                raise StructuredError(
                    f"`type_expr` must have an element type expression for `{type_}`."
                )

            output = []

            for idx, elem in enumerate(obj):
                try:
                    elem = self.structure(elem, type_args[0])
                except StructuredError as ex:
                    raise StructuredError(
                        f"The element at index {idx} in the sequence cannot be structured. See nested exception for details."
                    ) from ex

                output.append(elem)

            return output

        raise StructuredError(
            f"`obj` must be of type `{Sequence}`, but is of type `{type(obj)}` instead."
        )

    @staticmethod
    def _structure_literal(
        type_: Any, type_args: tuple[Any, ...], obj: object, set_empty: bool
    ) -> str:
        if isinstance(obj, str):
            if obj in type_args:
                return obj

            values = ", ".join(type_args)

            raise StructuredError(
                f"`obj` must be one of the following values, but is '{obj}' instead: {values}"
            )

        raise StructuredError(
            f"`obj` must be of type `{str}`, but is of type `{type(obj)}` instead."
        )

    @staticmethod
    def _structure_path(
        type_: Any, type_args: tuple[Any, ...], obj: object, set_empty: bool
    ) -> Path:
        if isinstance(obj, Path):
            return obj

        if isinstance(obj, str):
            return Path(obj)

        raise StructuredError(
            f"`obj` must be of type `{Path}` or `{str}`, but is of type `{type(obj)}` instead."
        )

    def _structure_set(
        self, type_: Any, type_args: tuple[Any, ...], obj: object, set_empty: bool
    ) -> set[object]:
        if isinstance(obj, set):
            if len(type_args) != 1:
                raise StructuredError(
                    f"`type_expr` must have an element type expression for `{type_}`."
                )

            return {self.structure(e, type_args[0]) for e in obj}

        if isinstance(obj, Sequence):
            if len(type_args) != 1:
                raise StructuredError(
                    f"`type_expr` must have an element type expression for `{type_}`."
                )

            tmp = [self.structure(e, type_args[0]) for e in obj]

            output = set(tmp)

            if len(output) != len(tmp):
                raise StructuredError(
                    f"All elements of `obj` must be unique to be treated as a `{set}`."
                )

            return output

        raise StructuredError(
            f"`obj` must be of type `{set}` or `{Sequence}`, but is of type `{type(obj)}` instead."
        )

    def _structure_tuple(
        self, type_: Any, type_args: tuple[Any, ...], obj: object, set_empty: bool
    ) -> tuple[object, ...]:
        if isinstance(obj, Sequence):
            num_args = len(type_args)

            if num_args == 0:
                raise StructuredError(
                    f"`type_expr` must have an element type expression for `{type_}`."
                )

            if num_args == 2 and type_args[1] is Ellipsis:  # homogeneous
                tmp = self._structure_list(type_, type_args[:1], obj, set_empty)

                return tuple(tmp)

            if len(obj) != num_args:  # heterogeneous
                raise StructuredError(
                    f"`obj` must be have {num_args} element(s), but it has {len(obj)} element(s)."
                )

            output = []

            for idx, elem in enumerate(obj):
                try:
                    elem = self.structure(elem, type_args[idx])
                except StructuredError as ex:
                    raise StructuredError(
                        f"The element at index {idx} in the sequence cannot be structured. See nested exception for details."
                    ) from ex

                output.append(elem)

            return tuple(output)

        raise StructuredError(
            f"`obj` must be of type `{tuple}` or `{Sequence}`, but is of type `{type(obj)}` instead."
        )

    def _structure_union(
        self, type_: Any, type_args: tuple[Any, ...], obj: object, set_empty: bool
    ) -> object:
        is_optional = len(type_args) == 2 and NoneType in type_args

        if is_optional and obj is None:
            return obj

        for type_expr in type_args:
            try:
                return self.structure(obj, type_expr, set_empty=set_empty)
            except StructuredError:
                if is_optional:
                    raise

                continue

        type_hints = ", ".join(str(t) for t in type_args)

        raise StructuredError(
            f"`obj` must be parseable as one of the following union elements: {type_hints}"
        )

    def unstructure(self, obj: object) -> object:
        type_ = type(obj)

        lookup_type: type

        if is_dataclass(type_):
            lookup_type = DataClass
        elif issubclass(type_, Enum):
            lookup_type = Enum
        elif issubclass(type_, Mapping):
            lookup_type = Mapping
        elif issubclass(type_, Path):
            lookup_type = Path
        elif issubclass(type_, Set):
            lookup_type = Set
        else:
            lookup_type = type_

        try:
            fn = self._unstructure_fns[lookup_type]
        except KeyError:
            supported_types = ", ".join(str(t) for t in self._unstructure_fns.keys())

            raise StructuredError(
                f"`obj` must be of one of the following types, but is `{type(obj)}` instead: {supported_types}"
            ) from None

        try:
            return fn(obj)
        except StructuredError as ex:
            raise StructuredError(
                "`obj` cannot be unstructured. See nested exception for details."
            ) from ex

    @staticmethod
    def _unstructure_identity(obj: Any) -> object:
        return obj

    def _unstructure_dataclass(self, obj: Any) -> dict[str, object]:
        type_ = type(obj)

        output: dict[str, object] = {}

        for field in fields(type_):
            value = getattr(obj, field.name)

            try:
                output[field.name] = self.unstructure(value)
            except StructuredError as ex:
                raise StructuredError(
                    f"The `{field.name}` field of the dataclass cannot be unstructured. See nested exception for details."
                ) from ex

        return output

    @staticmethod
    def _unstructure_dtype(obj: Any) -> str:
        return str(obj)[6:]  # strip 'torch.'

    @staticmethod
    def _unstructure_device(obj: Any) -> str:
        return str(obj)

    @staticmethod
    def _unstructure_enum(obj: Any) -> str:
        return cast(str, obj.name)

    def _unstructure_mapping(self, obj: Any) -> dict[object, object]:
        output = {}

        for k, v in obj.items():
            k = self.unstructure(k)
            v = self.unstructure(v)

            output[k] = self.unstructure(v)

        return output

    @staticmethod
    def _unstructure_path(obj: Any) -> str:
        return str(obj)

    def _unstructure_sequence(self, obj: Any) -> list[object]:
        output = []

        for idx, elem in enumerate(obj):
            try:
                elem = self.unstructure(elem)
            except StructuredError as ex:
                raise StructuredError(
                    f"The element at index {idx} in the sequence cannot be unstructured. See nested exception for details."
                ) from ex

            output.append(elem)

        return output

    def _unstructure_set(self, obj: Any) -> list[object]:
        return [self.unstructure(e) for e in obj]


def register_objects(container: DependencyContainer) -> None:
    container.register_instance(ValueConverter, ValueConverter())


def get_value_converter() -> ValueConverter:
    return resolve(ValueConverter)  # type: ignore[no-any-return]


def is_unstructured(obj: object) -> bool:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not is_unstructured(k):
                return False

            if not is_unstructured(v):
                return False

        return True

    if isinstance(obj, list):
        for e in obj:
            if not is_unstructured(e):
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
    if isinstance(source, dict):
        if not isinstance(target, dict):
            target = {}

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

        for k, v in target.items():
            if k not in ignored_keys:
                output[k] = deepcopy(v)

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

        set_keys = source.get("_set_")
        if set_keys is not None:
            if not isinstance(set_keys, dict):
                raise StructuredError(
                    f"'{path}{sep}_set_' in `source` must be of type `dict`, but is of type `{type(set_keys).__name__}` instead."
                )

            for idx, (set_key, value) in enumerate(set_keys.items()):
                if not isinstance(set_key, str):
                    raise StructuredError(
                        f"Each key under '{path}{sep}_set_' in `source` must be of type `str`, but the key at index {idx} is of type `{type(set_key).__name__}` instead."
                    )

                if set_key not in output:
                    sub_path = set_key if not path else f"{path}.{set_key}"

                    raise StructuredError(
                        f"`target` must contain a '{sub_path}' key since it exists in `source`."
                    ) from None

                output[set_key] = deepcopy(value)

        for key, source_value in source.items():
            if key == "_del_" or key == "_add_" or key == "_set_":
                continue

            # Maintain backwards compatibility with older configuration API.
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

    if isinstance(source, list | dict):
        return deepcopy(source)

    return source


class StructuredError(ValueError):
    """Raised when a structure or unstructure operation fails."""
