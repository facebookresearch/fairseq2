# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import NoneType, UnionType
from typing import (
    Any,
    Literal,
    Protocol,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import torch

from fairseq2.typing import EMPTY, DataClass, DataType, Device


class _Structurer(Protocol):
    def __call__(
        self,
        orig_type: object,
        type_args: tuple[object, ...],
        obj: object,
        set_empty: bool,
    ) -> object: ...


class _Unstructurer(Protocol):
    def __call__(self, obj: object) -> object: ...


class ValueConverter:
    """Structures objects using provided type expressions."""

    _structurers: dict[object, _Structurer]
    _unstructurers: dict[type, _Unstructurer]

    def __init__(self) -> None:
        self._structurers = {
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

        self._unstructurers = {
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

    def structure(self, obj: object, type_: object, *, set_empty: bool = False) -> Any:
        orig_type, type_args = get_origin(type_), get_args(type_)

        if orig_type is None:
            orig_type = type_

        if orig_type is object or orig_type is Any:
            return obj

        lookup_type = orig_type

        if isinstance(orig_type, type):
            if is_dataclass(orig_type):
                lookup_type = DataClass
            elif issubclass(orig_type, Enum):
                lookup_type = Enum

        structurer = self._structurers.get(lookup_type)
        if structurer is None:
            supported_types = ", ".join(str(t) for t in self._structurers.keys())

            raise StructureError(
                f"`type_` must be the value of a type expression consisting of the following types, but is `{type_}` instead: {supported_types}"
            ) from None

        try:
            return structurer(orig_type, type_args, obj, set_empty)
        except StructureError as ex:
            raise StructureError(
                f"`obj` cannot be structured to `{type_}`. See the nested exception for details."
            ) from ex

    @staticmethod
    def _structure_primitive(
        orig_type: object, type_args: tuple[object, ...], obj: object, set_empty: bool
    ) -> object:
        kls = cast(type, orig_type)

        if isinstance(obj, kls):
            return obj

        try:
            return kls(obj)
        except (TypeError, ValueError) as ex:
            raise StructureError(
                f"`obj` cannot be parsed as `{kls}`. See the nested exception for details."
            ) from ex

    @staticmethod
    def _structure_identity(
        orig_type: object, type_args: tuple[object, ...], obj: object, set_empty: bool
    ) -> object:
        kls = cast(type, orig_type)

        if isinstance(obj, kls):
            return obj

        raise StructureError(
            f"`obj` must be of type `{kls}`, but is of type `{type(obj)}` instead."
        )

    def _structure_dataclass(
        self,
        orig_type: object,
        type_args: tuple[object, ...],
        obj: object,
        set_empty: bool,
    ) -> object:
        kls = cast(type[DataClass], orig_type)

        if kls is DataClass:
            raise StructureError(
                f"`type` must be a concrete dataclass type, but is `{DataClass}` instead."
            )

        if isinstance(obj, kls):
            values = {f.name: getattr(obj, f.name) for f in fields(kls)}

            return self._create_dataclass(kls, values, set_empty)

        if isinstance(obj, Mapping):
            values = self.structure(obj, dict[str, object])

            return self._create_dataclass(kls, values, set_empty)

        raise StructureError(
            f"`obj` must be of type `{kls}` or `{Mapping}`, but is of type `{type(obj)}` instead."
        )

    def _create_dataclass(
        self, kls: type[DataClass], values: dict[str, object], set_empty: bool
    ) -> object:
        type_hints = get_type_hints(kls)

        kwargs = {}

        for field in fields(kls):
            value = values.pop(field.name, EMPTY)

            # Fields with `init=False` are initialized in `__post_init__()`.
            if not field.init:
                continue

            if value is EMPTY:
                if not set_empty:
                    if field.default == MISSING and field.default_factory == MISSING:
                        raise StructureError(
                            f"The `{field.name}` field has no default value or factory."
                        )

                    continue

                if hasattr(kls, "__post_init__"):
                    raise StructureError(
                        f"The `{field.name}` field must not be `EMPTY` since `{kls}` has a `__post_init__()` method."
                    )
            else:
                try:
                    value = self.structure(
                        value, type_hints[field.name], set_empty=set_empty
                    )
                except StructureError as ex:
                    raise StructureError(
                        f"The `{field.name}` field cannot be structured. See the nested exception for details."
                    ) from ex

            kwargs[field.name] = value

        if values:
            extra_keys = ", ".join(sorted(values.keys()))

            raise StructureError(
                f"`obj` must contain only keys corresponding to the fields of `{kls}`, but it contains the following extra keys: {extra_keys}"
            )

        try:
            return kls(**kwargs)
        except TypeError as ex:
            raise StructureError(
                "The dataclass has one or more `InitVar` pseudo fields and cannot be constructed."
            ) from ex

    def _structure_dict(
        self,
        orig_type: object,
        type_args: tuple[object, ...],
        obj: object,
        set_empty: bool,
    ) -> dict[object, object]:
        if isinstance(obj, Mapping):
            if len(type_args) != 2:
                raise StructureError(
                    f"`type_` must have a key-value type annotation for `{orig_type}`."
                )

            output = {}

            for k, v in obj.items():
                k = self.structure(k, type_args[0])
                v = self.structure(v, type_args[1])

                output[k] = v

            return output

        raise StructureError(
            f"`obj` must be of type `{Mapping}`, but is of type `{type(obj)}` instead."
        )

    @staticmethod
    def _structure_dtype(
        orig_type: object, type_args: tuple[object, ...], obj: object, set_empty: bool
    ) -> DataType:
        if isinstance(obj, DataType):
            return obj

        if isinstance(obj, str):
            if obj.startswith("torch."):
                obj = obj[6:]

            if isinstance(dtype := getattr(torch, obj, None), DataType):
                return dtype

            raise StructureError(
                f"`obj` must be a `torch.dtype` identifier, but is '{obj}' instead."
            )

        raise StructureError(
            f"`obj` must be of type `{DataType}` or `{str}`, but is of type `{type(obj)}` instead."
        )

    @staticmethod
    def _structure_device(
        orig_type: object, type_args: tuple[object, ...], obj: object, set_empty: bool
    ) -> Device:
        if isinstance(obj, Device):
            return obj

        if isinstance(obj, str):
            try:
                return Device(obj)
            except RuntimeError as ex:
                raise StructureError(str(ex))

        raise StructureError(
            f"`obj` must be of type `{Device}` or `{str}`, but is of type `{type(obj)}` instead."
        )

    @staticmethod
    def _structure_enum(
        orig_type: object, type_args: tuple[object, ...], obj: object, set_empty: bool
    ) -> object:
        kls = cast(type[Enum], orig_type)

        if isinstance(obj, kls):
            return obj

        if isinstance(obj, str):
            try:
                return kls[obj]
            except KeyError:
                pass

            values = ", ".join(e.name for e in kls)

            raise StructureError(
                f"`obj` must be one of the following enumeration values, but is '{obj}' instead: {values}"
            ) from None

        raise StructureError(
            f"`obj` must be of type `{kls}` or `{str}`, but is of type `{type(obj)}` instead."
        )

    def _structure_list(
        self,
        orig_type: object,
        type_args: tuple[object, ...],
        obj: object,
        set_empty: bool,
    ) -> list[object]:
        if isinstance(obj, Sequence):
            if len(type_args) != 1:
                raise StructureError(
                    f"`type_` must have an element type annotation for `{orig_type}`."
                )

            output = []

            for idx, elem in enumerate(obj):
                try:
                    elem = self.structure(elem, type_args[0])
                except StructureError as ex:
                    raise StructureError(
                        f"The element at index {idx} in the sequence cannot be structured. See the nested exception for details."
                    ) from ex

                output.append(elem)

            return output

        raise StructureError(
            f"`obj` must be of type `{Sequence}`, but is of type `{type(obj)}` instead."
        )

    @staticmethod
    def _structure_literal(
        orig_type: object, type_args: tuple[object, ...], obj: object, set_empty: bool
    ) -> str:
        if isinstance(obj, str):
            if obj in type_args:
                return obj

            values = ", ".join(str(t) for t in type_args)

            raise StructureError(
                f"`obj` must be one of the following values, but is '{obj}' instead: {values}"
            )

        raise StructureError(
            f"`obj` must be of type `{str}`, but is of type `{type(obj)}` instead."
        )

    @staticmethod
    def _structure_path(
        orig_type: object, type_args: tuple[object, ...], obj: object, set_empty: bool
    ) -> Path:
        if isinstance(obj, Path):
            return obj

        if isinstance(obj, str):
            return Path(obj)

        raise StructureError(
            f"`obj` must be of type `{Path}` or `{str}`, but is of type `{type(obj)}` instead."
        )

    def _structure_set(
        self,
        orig_type: object,
        type_args: tuple[object, ...],
        obj: object,
        set_empty: bool,
    ) -> set[object]:
        if isinstance(obj, set):
            if len(type_args) != 1:
                raise StructureError(
                    f"`type_` must have an element type annotation for `{orig_type}`."
                )

            return {self.structure(e, type_args[0]) for e in obj}

        if isinstance(obj, Sequence):
            if len(type_args) != 1:
                raise StructureError(
                    f"`type_` must have an element type annotation for `{orig_type}`."
                )

            tmp = [self.structure(e, type_args[0]) for e in obj]

            output = set(tmp)

            if len(output) != len(tmp):
                raise StructureError(
                    f"All elements of `obj` must be unique to be treated as a `{set}`."
                )

            return output

        raise StructureError(
            f"`obj` must be of type `{set}` or `{Sequence}`, but is of type `{type(obj)}` instead."
        )

    def _structure_tuple(
        self,
        orig_type: object,
        type_args: tuple[object, ...],
        obj: object,
        set_empty: bool,
    ) -> tuple[object, ...]:
        if isinstance(obj, Sequence):
            num_args = len(type_args)

            if num_args == 0:
                raise StructureError(
                    f"`type_` must have an element type annotation for `{orig_type}`."
                )

            if num_args == 2 and type_args[1] is Ellipsis:  # homogeneous
                tmp = self._structure_list(orig_type, type_args[:1], obj, set_empty)

                return tuple(tmp)

            if len(obj) != num_args:  # heterogeneous
                raise StructureError(
                    f"`obj` must be have {num_args} element(s), but it has {len(obj)} element(s)."
                )

            output = []

            for idx, elem in enumerate(obj):
                try:
                    elem = self.structure(elem, type_args[idx])
                except StructureError as ex:
                    raise StructureError(
                        f"The element at index {idx} in the sequence cannot be structured. See the nested exception for details."
                    ) from ex

                output.append(elem)

            return tuple(output)

        raise StructureError(
            f"`obj` must be of type `{tuple}` or `{Sequence}`, but is of type `{type(obj)}` instead."
        )

    def _structure_union(
        self,
        orig_type: object,
        type_args: tuple[object, ...],
        obj: object,
        set_empty: bool,
    ) -> object:
        is_optional = len(type_args) == 2 and NoneType in type_args

        if is_optional and obj is None:
            return obj

        for type_ in type_args:
            try:
                return self.structure(obj, type_, set_empty=set_empty)
            except StructureError:
                if is_optional:
                    raise

                continue

        types = ", ".join(str(t) for t in type_args)

        raise StructureError(
            f"`obj` must be parseable as one of the following union elements: {types}"
        )

    def unstructure(self, obj: object) -> object:
        kls = type(obj)

        lookup_kls: type[object]

        if is_dataclass(kls):
            lookup_kls = DataClass
        elif issubclass(kls, Enum):
            lookup_kls = Enum
        elif issubclass(kls, Mapping):
            lookup_kls = Mapping
        elif issubclass(kls, Path):
            lookup_kls = Path
        elif issubclass(kls, Set):
            lookup_kls = Set
        else:
            lookup_kls = kls

        unstructurer = self._unstructurers.get(lookup_kls)
        if unstructurer is None:
            supported_types = ", ".join(str(t) for t in self._unstructurers.keys())

            raise StructureError(
                f"`obj` must be of one of the following types, but is of type `{type(obj)}` instead: {supported_types}"
            ) from None

        try:
            return unstructurer(obj)
        except StructureError as ex:
            raise StructureError(
                "`obj` cannot be unstructured. See the nested exception for details."
            ) from ex

    @staticmethod
    def _unstructure_identity(obj: object) -> object:
        return obj

    def _unstructure_dataclass(self, obj: object) -> dict[str, object]:
        d = cast(DataClass, obj)

        kls = type(d)

        output: dict[str, object] = {}

        for field in fields(kls):
            value = getattr(obj, field.name)

            try:
                output[field.name] = self.unstructure(value)
            except StructureError as ex:
                raise StructureError(
                    f"The `{field.name}` field cannot be unstructured. See the nested exception for details."
                ) from ex

        return output

    @staticmethod
    def _unstructure_dtype(obj: object) -> str:
        return str(obj)[6:]  # strip 'torch.'

    @staticmethod
    def _unstructure_device(obj: object) -> str:
        return str(obj)

    @staticmethod
    def _unstructure_enum(obj: object) -> str:
        return cast(Enum, obj).name

    def _unstructure_mapping(self, obj: object) -> dict[object, object]:
        output = {}

        m = cast(Mapping[object, object], obj)

        for k, v in m.items():
            k = self.unstructure(k)
            v = self.unstructure(v)

            output[k] = self.unstructure(v)

        return output

    @staticmethod
    def _unstructure_path(obj: object) -> str:
        return str(obj)

    def _unstructure_sequence(self, obj: object) -> list[object]:
        output = []

        s = cast(Sequence[object], obj)

        for idx, elem in enumerate(s):
            try:
                elem = self.unstructure(elem)
            except StructureError as ex:
                raise StructureError(
                    f"The element at index {idx} in the sequence cannot be unstructured. See the nested exception for details."
                ) from ex

            output.append(elem)

        return output

    def _unstructure_set(self, obj: object) -> list[object]:
        s = cast(set[object], obj)

        return [self.unstructure(e) for e in s]


class StructureError(ValueError):
    """Raised when a structure or unstructure operation fails."""


default_value_converter = ValueConverter()


T = TypeVar("T")


def structure(obj: object, kls: type[T], *, set_empty: bool = False) -> T:
    obj = default_value_converter.structure(obj, kls, set_empty=set_empty)

    return cast(T, obj)


def unstructure(obj: object) -> object:
    return default_value_converter.unstructure(obj)


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
