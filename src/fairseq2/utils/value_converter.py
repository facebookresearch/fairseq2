# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from enum import Enum
from pathlib import Path, PosixPath
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Sequence,
    Set,
    Type,
    Union,
    cast,
    get_args,
    get_origin,
)

import torch

from fairseq2.typing import DataType, Device


class ValueConverter:
    """Structures and unstructures objects using provided type hints."""

    _structure_fns: Dict[object, Callable[[Any, Any, Any], Any]]
    _unstructure_fns: Dict[object, Callable[[Any, Any], Any]]

    def __init__(self) -> None:
        NoneType = type(None)

        self._structure_fns = {
            # fmt: off
            bool:      self._structure_identity,
            DataType:  self._structure_dtype,
            Device:    self._structure_device,
            dict:      self._structure_dict,
            float:     self._structure_primitive,
            Enum:      self._structure_enum,
            int:       self._structure_primitive,
            list:      self._structure_list,
            Literal:   self._structure_literal,
            NoneType:  self._structure_identity,
            Path:      self._structure_path,
            PosixPath: self._structure_path,
            set:       self._structure_set,
            str:       self._structure_identity,
            tuple:     self._structure_tuple,
            Union:     self._structure_union,
            # fmt: on
        }

        self._unstructure_fns = {
            # fmt: off
            bool:      self._unstructure_identity,
            DataType:  self._unstructure_dtype,
            Device:    self._unstructure_device,
            dict:      self._unstructure_dict,
            float:     self._unstructure_identity,
            Enum:      self._unstructure_enum,
            int:       self._unstructure_identity,
            list:      self._unstructure_sequence,
            NoneType:  self._unstructure_identity,
            Path:      self._unstructure_path,
            PosixPath: self._unstructure_path,
            set:       self._unstructure_set,
            str:       self._unstructure_identity,
            tuple:     self._unstructure_sequence,
            # fmt: on
        }

    def structure(self, obj: Any, type_hint: Any) -> Any:
        """
        :param obj:
            The object to structure based on ``type_hint``.
        :param type_hint:
            The type hint. Typically retrieved via ``typing.get_type_hints()``.
        """
        kls, kls_args = get_origin(type_hint), get_args(type_hint)

        if kls is None:
            kls = type_hint

        if kls is Any:
            return obj

        if isinstance(kls, type):
            lookup_kls = Enum if issubclass(kls, Enum) else kls
        else:
            lookup_kls = kls  # typing special form

        try:
            fn = self._structure_fns[lookup_kls]
        except KeyError:
            supported = ", ".join(str(t) for t in self._structure_fns.keys())

            raise ValueError(
                f"`type_hint` of `obj` must be of one of the following, but is `{type_hint}` instead: {supported}"
            )

        try:
            return fn(kls, kls_args, obj)
        except (TypeError, ValueError) as ex:
            raise TypeError(
                f"`obj` cannot be structured to type `{type_hint}`. See nested exception for details."
            ) from ex

    @staticmethod
    def _structure_identity(kls: Any, kls_args: Any, obj: Any) -> Any:
        if isinstance(obj, kls):
            return obj

        raise TypeError(
            f"`obj` must be of type `{kls}`, but is of type `{type(obj)}` instead."
        )

    @staticmethod
    def _structure_primitive(kls: Any, kls_args: Any, obj: Any) -> Any:
        if isinstance(obj, kls):
            return obj

        return kls(obj)

    @staticmethod
    def _structure_dtype(kls: Any, kls_args: Any, obj: Any) -> Any:
        if isinstance(obj, DataType):
            return obj

        if isinstance(obj, str):
            if obj.startswith("torch."):
                obj = obj[6:]

            if isinstance(dtype := getattr(torch, obj, None), DataType):
                return dtype

            raise ValueError(
                f"`obj` must be a `torch.dtype` identifier, but is '{obj}' instead."
            )

        raise TypeError(
            f"`obj` must be of type `{DataType}` or `{str}`, but is of type `{type(obj)}` instead."
        )

    @staticmethod
    def _structure_device(kls: Any, kls_args: Any, obj: Any) -> Any:
        if isinstance(obj, Device):
            return obj

        if isinstance(obj, str):
            try:
                return Device(obj)
            except RuntimeError as ex:
                raise ValueError(str(ex))

        raise TypeError(
            f"`obj` must be of type `{Device}` or `{str}`, but is of type `{type(obj)}` instead."
        )

    def _structure_dict(self, kls: Any, kls_args: Any, obj: Any) -> Any:
        if isinstance(obj, dict):
            if len(kls_args) != 2:
                raise TypeError("`type_hint` has no type annotation for `dict`.")

            output = {}

            for k, v in obj.items():
                k = self.structure(k, kls_args[0])
                v = self.structure(v, kls_args[1])

                output[k] = v

            return output

        raise TypeError(
            f"`obj` must be of type `{dict}`, but is of type `{type(obj)}` instead."
        )

    @staticmethod
    def _structure_enum(kls: Any, kls_args: Any, obj: Any) -> Any:
        enum_kls = cast(Type[Enum], kls)

        if isinstance(obj, enum_kls):
            return obj

        if isinstance(obj, str):
            try:
                return enum_kls[obj]  # type: ignore[index]
            except KeyError:
                raise ValueError(
                    f"`obj` must be one of the following enumeration values, but is '{obj}' instead: {', '.join(e.name for e in enum_kls)}."
                )

        raise TypeError(
            f"`obj` must be of type `{kls}` or `{str}`, but is of type `{type(obj)}` instead."
        )

    def _structure_list(self, kls: Any, kls_args: Any, obj: Any) -> Any:
        if isinstance(obj, Sequence):
            if len(kls_args) != 1:
                raise TypeError("`type_hint` has no type annotation for `list`.")

            return [self.structure(e, kls_args[0]) for e in obj]

        raise TypeError(
            f"`obj` must be of type `{Sequence}`, but is of type `{type(obj)}` instead."
        )

    @staticmethod
    def _structure_literal(kls: Any, kls_args: Any, obj: Any) -> Any:
        if isinstance(obj, str):
            if obj in kls_args:
                return obj

            raise ValueError(
                f"`obj` must be one of the following values, but is '{obj}' instead: {', '.join(kls_args)}."
            )

        raise TypeError(
            f"`obj` must be of type `{str}`, but is of type `{type(obj)}` instead."
        )

    @staticmethod
    def _structure_path(kls: Any, kls_args: Any, obj: Any) -> Any:
        if isinstance(obj, Path):
            return obj

        if isinstance(obj, str):
            return Path(obj)

        raise TypeError(
            f"`obj` must be of type `{Path}` or `{str}`, but is of type `{type(obj)}` instead."
        )

    def _structure_set(self, kls: Any, kls_args: Any, obj: Any) -> Any:
        if isinstance(obj, Set):
            if len(kls_args) != 1:
                raise TypeError("`type_hint` has no type annotation for `set`.")

            return {self.structure(e, kls_args[0]) for e in obj}

        if isinstance(obj, Sequence):
            if len(kls_args) != 1:
                raise TypeError("`type_hint` has no type annotation for `set`.")

            tmp = [self.structure(e, kls_args[0]) for e in obj]

            output = set(tmp)

            if len(output) != len(tmp):
                raise ValueError(
                    f"All elements of `obj` must be unique to be treated as a `{set}`."
                )

            return output

        raise TypeError(
            f"`obj` must be of type `{set}` or `{Sequence}`, but is of type `{type(obj)}` instead."
        )

    def _structure_tuple(self, kls: Any, kls_args: Any, obj: Any) -> Any:
        if isinstance(obj, Sequence):
            num_args = len(kls_args)

            if num_args == 0:
                raise TypeError("`type_hint` has no type annotation for `tuple`.")

            if num_args == 2 and kls_args[1] is Ellipsis:  # homogeneous
                tmp = [self.structure(e, kls_args[0]) for e in obj]

                return tuple(tmp)

            if len(obj) != num_args:  # heterogeneous
                raise TypeError(
                    f"`obj` must be have {num_args} elements, but it has {len(obj)} elements."
                )

            output = []

            for i, e in enumerate(obj):
                output.append(self.structure(e, kls_args[i]))

            return tuple(output)

        raise TypeError(
            f"`obj` must be of type `{tuple}` or `{Sequence}`, but is of type `{type(obj)}` instead."
        )

    def _structure_union(self, kls: Any, kls_args: Any, obj: Any) -> Any:
        for kls_ in kls_args:
            try:
                return self.structure(obj, kls_)
            except (TypeError, ValueError):
                continue

        raise TypeError(
            f"`obj` must be parseable as one of the following union types: {', '.join(str(t) for t in kls_args)}"
        )

    def unstructure(self, obj: Any) -> Any:
        kls = type(obj)

        lookup_kls = Enum if issubclass(kls, Enum) else kls

        try:
            fn = self._unstructure_fns[lookup_kls]
        except KeyError:
            supported_types = ", ".join(str(t) for t in self._unstructure_fns.keys())

            raise TypeError(
                f"`obj` must be of one of the following types, but is of type `{type(obj)}` instead: {supported_types}"
            )

        return fn(kls, obj)

    @staticmethod
    def _unstructure_identity(kls: Any, obj: Any) -> Any:
        return obj

    @staticmethod
    def _unstructure_dtype(kls: Any, obj: Any) -> Any:
        return str(obj)[6:]  # strip 'torch.'

    @staticmethod
    def _unstructure_device(kls: Any, obj: Any) -> Any:
        return str(obj)

    def _unstructure_dict(self, kls: Any, obj: Any) -> Any:
        return {self.unstructure(k): self.unstructure(v) for k, v in obj.items()}

    def _unstructure_enum(self, kls: Any, obj: Any) -> Any:
        return obj.name

    def _unstructure_set(self, kls: Any, obj: Any) -> Any:
        return [self.unstructure(e) for e in obj]

    def _unstructure_sequence(self, kls: Any, obj: Any) -> Any:
        return [self.unstructure(e) for e in obj]

    @staticmethod
    def _unstructure_path(kls: Any, obj: Any) -> Any:
        return str(obj)


default_value_converter = ValueConverter()
