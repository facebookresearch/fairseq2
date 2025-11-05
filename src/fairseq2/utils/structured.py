# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence, Set
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import NoneType, UnionType
from typing import (
    Any,
    Literal,
    Protocol,
    Union,
    cast,
    final,
    get_args,
    get_origin,
    get_type_hints,
)

import torch
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.typing import DataClass


class ValueConverter(ABC):
    @abstractmethod
    def structure(self, obj: object, target_type: object) -> Any: ...

    @abstractmethod
    def unstructure(self, obj: object) -> object: ...


class StructureError(Exception):
    """Raised when a structure or unstructure operation fails."""


class _Structurer(Protocol):
    def __call__(
        self, origin_type: object, type_args: tuple[object, ...], obj: object
    ) -> object: ...


class _Unstructurer(Protocol):
    def __call__(self, obj: object) -> object: ...


@final
class StandardValueConverter(ValueConverter):
    def __init__(self) -> None:
        self._structurers: dict[object, _Structurer] = {
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

        self._unstructurers: dict[type[object], _Unstructurer] = {
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

    @override
    def structure(self, obj: object, target_type: object) -> Any:
        origin_type, type_args = get_origin(target_type), get_args(target_type)

        if origin_type is None:
            origin_type = target_type

        if origin_type is object:
            return obj

        lookup_type = origin_type

        if isinstance(origin_type, type):
            if is_dataclass(origin_type):
                lookup_type = DataClass
            elif issubclass(origin_type, Enum):
                lookup_type = Enum

        structurer = self._structurers.get(lookup_type)
        if structurer is None:
            s = ", ".join(str(t) for t in self._structurers.keys())

            raise ValueError(
                f"`target_type` must represent a type expression consisting of supported types, but is `{target_type}` instead. Supported types are {s}"
            )

        try:
            return structurer(origin_type, type_args, obj)
        except StructureError as ex:
            raise StructureError(
                f"Value cannot be structured to `{target_type}`."
            ) from ex

    def _structure_primitive(
        self, origin_type: object, type_args: tuple[object, ...], obj: object
    ) -> object:
        kls = cast(type, origin_type)

        if isinstance(obj, kls):
            return obj

        try:
            return kls(obj)
        except (TypeError, ValueError) as ex:
            raise StructureError(f"`obj` cannot be parsed as `{kls}`.") from ex

    def _structure_identity(
        self, origin_type: object, type_args: tuple[object, ...], obj: object
    ) -> object:
        kls = cast(type, origin_type)

        if isinstance(obj, kls):
            return obj

        raise StructureError(
            f"Value must be of type `{kls}`, but is of type `{type(obj)}` instead."
        )

    def _structure_dataclass(
        self, origin_type: object, type_args: tuple[object, ...], obj: object
    ) -> object:
        kls = cast(type[DataClass], origin_type)

        if kls is DataClass:
            raise ValueError(
                f"`type` must be a concrete dataclass type, but is `{DataClass}` instead."
            )

        if isinstance(obj, kls):
            return obj

        if isinstance(obj, Mapping):
            values = self.structure(obj, dict[str, object])

            return self._create_dataclass(kls, values)

        raise StructureError(
            f"Value must be of type `{kls}` or `{Mapping}`, but is of type `{type(obj)}` instead."
        )

    def _create_dataclass(
        self, kls: type[DataClass], values: dict[str, object]
    ) -> object:
        type_hints = get_type_hints(kls)

        kwargs = {}

        empty_sentinel = object()

        for field in fields(kls):
            value = values.pop(field.name, empty_sentinel)

            # Fields with `init=False` are initialized in `__post_init__()`.
            if not field.init:
                continue

            if value is empty_sentinel:
                if field.default == MISSING and field.default_factory == MISSING:
                    raise StructureError(
                        f"`{field.name}` field has no default value or factory."
                    )

                continue
            else:
                try:
                    value = self.structure(value, type_hints[field.name])
                except StructureError as ex:
                    raise StructureError(
                        f"`{field.name}` field cannot be structured."
                    ) from ex

            kwargs[field.name] = value

        if values:
            extra_keys = ", ".join(sorted(values.keys()))

            raise StructureError(
                f"Value must contain only keys corresponding to the fields of `{kls}`, but it contains extra keys {extra_keys}."
            )

        try:
            return kls(**kwargs)
        except TypeError as ex:
            raise StructureError(
                "dataclass has one or more `InitVar` pseudo fields and cannot be constructed."
            ) from ex

    def _structure_dict(
        self, origin_type: object, type_args: tuple[object, ...], obj: object
    ) -> dict[object, object]:
        if isinstance(obj, Mapping):
            if len(type_args) != 2:
                raise ValueError(
                    f"`target_type` must have a key-value type annotation for `{origin_type}`."
                )

            output = {}

            for k, v in obj.items():
                try:
                    k = self.structure(k, type_args[0])
                except StructureError as ex:
                    raise StructureError(f"{k} key cannot be structured.") from ex

                try:
                    v = self.structure(v, type_args[1])
                except StructureError as ex:
                    raise StructureError(
                        f"Value of the {k} key cannot be structured."
                    ) from ex

                output[k] = v

            return output

        raise StructureError(
            f"Value must be of type `{Mapping}`, but is of type `{type(obj)}` instead."
        )

    def _structure_dtype(
        self, origin_type: object, type_args: tuple[object, ...], obj: object
    ) -> DataType:
        if isinstance(obj, DataType):
            return obj

        if isinstance(obj, str):
            if obj.startswith("torch."):
                obj = obj[6:]

            if isinstance(dtype := getattr(torch, obj, None), DataType):
                return dtype

            raise StructureError(
                f"Value must be a `torch.dtype` identifier, but is {obj} instead."
            )

        raise StructureError(
            f"Value must be of type `{DataType}` or `{str}`, but is of type `{type(obj)}` instead."
        )

    def _structure_device(
        self, origin_type: object, type_args: tuple[object, ...], obj: object
    ) -> Device:
        if isinstance(obj, Device):
            return obj

        if isinstance(obj, str):
            try:
                return Device(obj)
            except RuntimeError as ex:
                raise StructureError(str(ex)) from None

        raise StructureError(
            f"Value must be of type `{Device}` or `{str}`, but is of type `{type(obj)}` instead."
        )

    def _structure_enum(
        self, origin_type: object, type_args: tuple[object, ...], obj: object
    ) -> object:
        kls = cast(type[Enum], origin_type)

        if isinstance(obj, kls):
            return obj

        if isinstance(obj, str):
            try:
                return kls[obj]
            except KeyError:
                pass

            values = ", ".join(e.name for e in kls)

            raise StructureError(
                f"Value must be a supported enumeration value, but is {obj} instead. Supported values are {values}."
            )

        raise StructureError(
            f"Value must be of type `{kls}` or `{str}`, but is of type `{type(obj)}` instead."
        )

    def _structure_list(
        self, origin_type: object, type_args: tuple[object, ...], obj: object
    ) -> list[object]:
        if isinstance(obj, Sequence):
            if len(type_args) != 1:
                raise ValueError(
                    f"`target_type` must have an element type expression for `{origin_type}`."
                )

            output = []

            for idx, elem in enumerate(obj):
                try:
                    elem = self.structure(elem, type_args[0])
                except StructureError as ex:
                    raise StructureError(
                        f"Element {elem} at index {idx} cannot be structured as {type_args[0]}."
                    ) from ex

                output.append(elem)

            return output

        raise StructureError(
            f"Value must be of type `{Sequence}`, but is of type `{type(obj)}` instead."
        )

    def _structure_literal(
        self, origin_type: object, type_args: tuple[object, ...], obj: object
    ) -> str:
        if isinstance(obj, str):
            if obj in type_args:
                return obj

            values = ", ".join(str(t) for t in type_args)

            raise StructureError(
                f"Value must be a supported literal value, but is {obj} instead. Supported values are {values}."
            )

        raise StructureError(
            f"Value must be of type `{str}`, but is of type `{type(obj)}` instead."
        )

    def _structure_path(
        self, origin_type: object, type_args: tuple[object, ...], obj: object
    ) -> Path:
        if isinstance(obj, Path):
            return obj

        if isinstance(obj, str):
            return Path(obj)

        raise StructureError(
            f"Value must be of type `{Path}` or `{str}`, but is of type `{type(obj)}` instead."
        )

    def _structure_set(
        self, origin_type: object, type_args: tuple[object, ...], obj: object
    ) -> set[object]:
        if isinstance(obj, set):
            if len(type_args) != 1:
                raise ValueError(
                    f"`target_type` must have an element type expression for `{origin_type}`."
                )

            try:
                return {self.structure(e, type_args[0]) for e in obj}
            except StructureError as ex:
                raise StructureError(
                    "One of the set elements cannot be structured."
                ) from ex

        if isinstance(obj, Sequence):
            if len(type_args) != 1:
                raise ValueError(
                    f"`target_type` must have an element type expression for `{origin_type}`."
                )

            tmp = []

            for idx, e in enumerate(obj):
                try:
                    e = self.structure(e, type_args[0])
                except StructureError as ex:
                    raise StructureError(
                        f"Element at index {idx} cannot be structured."
                    ) from ex

                tmp.append(e)

            output = set(tmp)

            if len(output) != len(tmp):
                raise StructureError(
                    f"All elements in the sequence must be unique to be treated as a `{set}`."
                )

            return output

        raise StructureError(
            f"Value must be of type `{set}` or `{Sequence}`, but is of type `{type(obj)}` instead."
        )

    def _structure_tuple(
        self, origin_type: object, type_args: tuple[object, ...], obj: object
    ) -> tuple[object, ...]:
        if isinstance(obj, Sequence):
            num_args = len(type_args)

            if num_args == 0:
                raise ValueError(
                    f"`target_type` must have an element type expression for `{origin_type}`."
                )

            if num_args == 2 and type_args[1] is Ellipsis:  # homogeneous
                tmp = self._structure_list(origin_type, type_args[:1], obj)

                return tuple(tmp)

            if len(obj) != num_args:  # heterogeneous
                raise StructureError(
                    f"Value must be have {num_args} element(s), but has {len(obj)} element(s) instead."
                )

            output = []

            for idx, elem in enumerate(obj):
                try:
                    elem = self.structure(elem, type_args[idx])
                except StructureError as ex:
                    raise StructureError(
                        f"Element at index {idx} cannot be structured."
                    ) from ex

                output.append(elem)

            return tuple(output)

        raise StructureError(
            f"Value must be of type `{tuple}` or `{Sequence}`, but is of type `{type(obj)}` instead."
        )

    def _structure_union(
        self, origin_type: object, type_args: tuple[object, ...], obj: object
    ) -> object:
        is_optional = len(type_args) == 2 and NoneType in type_args

        if is_optional and obj is None:
            return obj

        errors = []

        for target_type in type_args:
            try:
                return self.structure(obj, target_type)
            except StructureError as ex:
                if is_optional:
                    raise

                errors.append(ex)

        traces = []

        # TODO: Consider using an ExceptionGroup after Python 3.10.
        for error in errors:
            trace = traceback.format_exception(error)

            traces.append("\n".join(trace))

        e = "\n************************************************\n\n".join(traces)

        s = ", ".join(str(t) for t in type_args)

        raise StructureError(
            f"Value must be parseable as one of union elements {s}, but is {obj} instead.\n\n{e}"
        )

    @override
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
            s = ", ".join(str(t) for t in self._unstructurers.keys())

            raise StructureError(
                f"Value must be of one of the supported types, but is of type `{type(obj)}` instead. Supported types are {s}."
            )

        try:
            return unstructurer(obj)
        except StructureError as ex:
            raise StructureError("Value cannot be unstructured.") from ex

    def _unstructure_identity(self, obj: object) -> object:
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
                    f"`{field.name}` field cannot be unstructured."
                ) from ex

        return output

    def _unstructure_dtype(self, obj: object) -> str:
        return str(obj)[6:]  # strip 'torch.'

    def _unstructure_device(self, obj: object) -> str:
        return str(obj)

    def _unstructure_enum(self, obj: object) -> str:
        return cast(Enum, obj).name

    def _unstructure_mapping(self, obj: object) -> dict[object, object]:
        output = {}

        m = cast(Mapping[object, object], obj)

        for k, v in m.items():
            k = self.unstructure(k)
            v = self.unstructure(v)

            output[k] = self.unstructure(v)

        return output

    def _unstructure_path(self, obj: object) -> str:
        return str(obj)

    def _unstructure_sequence(self, obj: object) -> list[object]:
        output = []

        s = cast(Sequence[object], obj)

        for idx, elem in enumerate(s):
            try:
                elem = self.unstructure(elem)
            except StructureError as ex:
                raise StructureError(
                    f"Element at index {idx} in the sequence cannot be unstructured."
                ) from ex

            output.append(elem)

        return output

    def _unstructure_set(self, obj: object) -> list[object]:
        s = cast(set[object], obj)

        return [self.unstructure(e) for e in s]
