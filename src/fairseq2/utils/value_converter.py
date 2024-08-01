# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import builtins
import sys
from dataclasses import fields, is_dataclass
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    Final,
    Literal,
    Mapping,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import torch

from fairseq2.typing import DataClass, DataType, Device, is_dataclass_instance
from fairseq2.utils.dataclass import EMPTY

_EMPTY_TAG: Final[str] = "~~"


NoneType = type(None)


class ValueConverter:
    """Structures and unstructures objects using provided type hints."""

    _structure_fns: Dict[object, Callable[[Any, Tuple[Any, ...], Any, bool], Any]]
    _unstructure_fns: Dict[object, Callable[[Any, Tuple[Any, ...], Any], Any]]

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
            NoneType:  self._structure_identity,
            Path:      self._structure_path,
            set:       self._structure_set,
            str:       self._structure_identity,
            tuple:     self._structure_tuple,
            Union:     self._structure_union,
            # fmt: on
        }

        self._unstructure_fns = {
            # fmt: off
            bool:        self._unstructure_identity,
            DataClass:   self._unstructure_dataclass,
            DataType:    self._unstructure_dtype,
            Device:      self._unstructure_device,
            float:       self._unstructure_identity,
            Enum:        self._unstructure_enum,
            int:         self._unstructure_identity,
            list:        self._unstructure_list,
            Literal:     self._unstructure_literal,
            Mapping:     self._unstructure_mapping,
            NoneType:    self._unstructure_identity,
            Path:        self._unstructure_path,
            AbstractSet: self._unstructure_set,
            str:         self._unstructure_identity,
            tuple:       self._unstructure_tuple,
            Union:       self._unstructure_union,
            # fmt: on
        }

        if sys.version_info >= (3, 10):
            from types import UnionType

            # Union types in PEP 604 (i.e. pipe) syntax use `types.UnionType`.
            self._structure_fns[UnionType] = self._structure_union

    def structure(self, obj: Any, type_hint: Any, *, set_empty: bool = False) -> Any:
        """
        :param obj:
            The object to structure based on ``type_hint``.
        :param type_hint:
            The type hint. Typically retrieved via ``typing.get_type_hints()``.
        :param set_empty:
            If ``True``, dataclass fields that are not present in ``obj`` will
            have their values set to ``EMPTY``.
        """
        kls, kls_args = get_origin(type_hint), get_args(type_hint)

        if kls is None:
            kls = type_hint

        if kls is Any:
            return obj

        lookup_kls = kls

        if isinstance(kls, type):
            if is_dataclass(kls):
                lookup_kls = DataClass
            elif issubclass(kls, Enum):
                lookup_kls = Enum
            elif issubclass(kls, Path):
                lookup_kls = Path

        try:
            fn = self._structure_fns[lookup_kls]
        except KeyError:
            supported_expr = ", ".join(str(t) for t in self._structure_fns.keys())

            raise ValueError(
                f"`type_hint` must be of one of the following expressions, but is `{type_hint}` instead: {supported_expr}"
            ) from None

        try:
            return fn(kls, kls_args, obj, set_empty)
        except (TypeError, ValueError, ImportError, AttributeError) as ex:
            raise TypeError(
                f"`obj` cannot be structured to `{type_hint}`. See nested exception for details."
            ) from ex

    @staticmethod
    def _structure_primitive(
        kls: Any, kls_args: Tuple[Any, ...], obj: Any, set_empty: bool
    ) -> Any:
        if isinstance(obj, kls):
            return obj

        return kls(obj)

    @staticmethod
    def _structure_identity(
        kls: Any, kls_args: Tuple[Any, ...], obj: Any, set_empty: bool
    ) -> Any:
        if isinstance(obj, kls):
            return obj

        raise TypeError(
            f"`obj` must be of type `{kls}`, but is of type `{type(obj)}` instead."
        )

    def _structure_dataclass(
        self, kls: Any, kls_args: Tuple[Any, ...], obj: Any, set_empty: bool
    ) -> Any:
        if kls is DataClass:
            return self._structure_unbounded_dataclass(kls, obj, set_empty)

        if isinstance(obj, kls):
            kls = type(obj)

            values = {f.name: getattr(obj, f.name) for f in fields(kls)}

            return self._create_dataclass(kls, values, set_empty=set_empty)

        if isinstance(obj, Mapping):
            values = self.structure(obj, Dict[str, Any])

            sub_kls_name = values.pop("_type_", None)
            if isinstance(sub_kls_name, str):
                sub_kls = self._str_to_type(sub_kls_name)

                if not issubclass(sub_kls, kls):
                    raise TypeError(f"`{sub_kls}` must be a subclass of `{kls}`.")

                kls = sub_kls

            return self._create_dataclass(kls, values, set_empty=set_empty)

        raise TypeError(
            f"`obj` must be of type `{kls}` or `{Mapping}`, but is of type `{type(obj)}` instead."
        )

    def _structure_unbounded_dataclass(
        self, kls: Any, obj: Any, set_empty: bool
    ) -> Any:
        if isinstance(obj, Mapping):
            values = self.structure(obj, Dict[str, Any])

            kls_name = values.pop("_type_", None)
            if not isinstance(kls_name, str):
                raise TypeError(
                    f"The mapping must contain a '_type_' key to structure an unbounded `{DataClass}`."
                )

            kls = self._str_to_type(kls_name)

            if not is_dataclass(kls):
                raise TypeError(f"`{kls}` must be a dataclass.")
        else:
            kls = type(obj)

            if not is_dataclass(kls):
                raise TypeError(f"`{kls}` must be a dataclass.")

            values = {f.name: getattr(obj, f.name) for f in fields(kls)}

        return self._create_dataclass(kls, values, set_empty=set_empty)

    def _create_dataclass(
        self, kls: Type[DataClass], values: Dict[str, Any], set_empty: bool
    ) -> DataClass:
        type_hints = get_type_hints(kls)

        kwargs = {}

        for field in fields(kls):
            try:
                value = values.pop(field.name)
            except KeyError:
                value = EMPTY

            # Fields with `init=False` are initialized in `__post_init__()`.
            if not field.init:
                continue

            if value == _EMPTY_TAG:
                value = EMPTY

            if value is not EMPTY:
                try:
                    value = self.structure(
                        value, type_hints[field.name], set_empty=set_empty
                    )
                except TypeError as ex:
                    raise TypeError(
                        f"'{field.name}' field cannot be structured to `{type_hints[field.name]}`. See nested exception for details."
                    ) from ex
            else:
                if set_empty:
                    if hasattr(kls, "__post_init__"):
                        raise TypeError(
                            f"`{kls}` has a `__post_init__()` method which is not supported when `set_empty` is set."
                        )
                else:
                    continue

            kwargs[field.name] = value

        if values:
            unknown_keys = ", ".join(list(values.keys()))

            raise ValueError(
                f"`obj` must not have the following keys which do not map to any field of `{kls}`: {unknown_keys}"
            )

        return kls(**kwargs)

    @staticmethod
    def _structure_dtype(
        kls: Any, kls_args: Tuple[Any, ...], obj: Any, set_empty: bool
    ) -> Any:
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
    def _structure_device(
        kls: Any, kls_args: Tuple[Any, ...], obj: Any, set_empty: bool
    ) -> Any:
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

    def _structure_dict(
        self, kls: Any, kls_args: Tuple[Any, ...], obj: Any, set_empty: bool
    ) -> Any:
        if isinstance(obj, Mapping):
            if len(kls_args) != 2:
                raise TypeError(f"`type_hint` has no type annotation for `{kls}`.")

            output = {}

            for k, v in obj.items():
                k = self.structure(k, kls_args[0])
                v = self.structure(v, kls_args[1])

                output[k] = v

            return output

        raise TypeError(
            f"`obj` must be of type `{Mapping}`, but is of type `{type(obj)}` instead."
        )

    @staticmethod
    def _structure_enum(
        kls: Any, kls_args: Tuple[Any, ...], obj: Any, set_empty: bool
    ) -> Any:
        if isinstance(obj, kls):
            return obj

        if isinstance(obj, str):
            try:
                return kls[obj]
            except KeyError:
                values = ", ".join(e.name for e in kls)

                raise ValueError(
                    f"`obj` must be one of the following enumeration values, but is '{obj}' instead: {values}"
                ) from None

        raise TypeError(
            f"`obj` must be of type `{kls}` or `{str}`, but is of type `{type(obj)}` instead."
        )

    def _structure_list(
        self, kls: Any, kls_args: Tuple[Any, ...], obj: Any, set_empty: bool
    ) -> Any:
        if isinstance(obj, Sequence):
            if len(kls_args) != 1:
                raise TypeError(f"`type_hint` has no type annotation for `{kls}`.")

            return [self.structure(e, kls_args[0]) for e in obj]

        raise TypeError(
            f"`obj` must be of type `{Sequence}`, but is of type `{type(obj)}` instead."
        )

    @staticmethod
    def _structure_literal(
        kls: Any, kls_args: Tuple[Any, ...], obj: Any, set_empty: bool
    ) -> Any:
        if isinstance(obj, str):
            if obj in kls_args:
                return obj

            values = ", ".join(kls_args)

            raise ValueError(
                f"`obj` must be one of the following values, but is '{obj}' instead: {values}"
            )

        raise TypeError(
            f"`obj` must be of type `{str}`, but is of type `{type(obj)}` instead."
        )

    @staticmethod
    def _structure_path(
        kls: Any, kls_args: Tuple[Any, ...], obj: Any, set_empty: bool
    ) -> Any:
        if isinstance(obj, Path):
            return obj

        if isinstance(obj, str):
            return Path(obj)

        raise TypeError(
            f"`obj` must be of type `{Path}` or `{str}`, but is of type `{type(obj)}` instead."
        )

    def _structure_set(
        self, kls: Any, kls_args: Tuple[Any, ...], obj: Any, set_empty: bool
    ) -> Any:
        if isinstance(obj, Set):
            if len(kls_args) != 1:
                raise TypeError(f"`type_hint` has no type annotation for `{kls}`.")

            return {self.structure(e, kls_args[0]) for e in obj}

        if isinstance(obj, Sequence):
            if len(kls_args) != 1:
                raise TypeError(f"`type_hint` has no type annotation for `{kls}`.")

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

    def _structure_tuple(
        self, kls: Any, kls_args: Tuple[Any, ...], obj: Any, set_empty: bool
    ) -> Any:
        if isinstance(obj, Sequence):
            num_args = len(kls_args)

            if num_args == 0:
                raise TypeError(f"`type_hint` has no type annotation for `{kls}`.")

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

    def _structure_union(
        self, kls: Any, kls_args: Tuple[Any, ...], obj: Any, set_empty: bool
    ) -> Any:
        is_optional = len(kls_args) == 2 and NoneType in kls_args

        if is_optional and obj is None:
            return obj

        for kls_ in kls_args:
            try:
                return self.structure(obj, kls_, set_empty=set_empty)
            except (TypeError, ValueError):
                if is_optional:
                    raise

                continue

        types = ", ".join(str(t) for t in kls_args)

        raise TypeError(
            f"`obj` must be parseable as one of the following union types: {types}"
        )

    def unstructure(self, obj: Any, type_hint: Any) -> Any:
        kls, kls_args = get_origin(type_hint), get_args(type_hint)

        if kls is None:
            kls = type_hint

        if kls is Any:
            return obj

        lookup_kls = kls

        if isinstance(kls, type):
            if is_dataclass(kls):
                lookup_kls = DataClass
            elif issubclass(kls, Enum):
                lookup_kls = Enum
            elif issubclass(kls, Mapping):
                lookup_kls = Mapping
            elif issubclass(kls, Path):
                lookup_kls = Path
            elif issubclass(kls, AbstractSet):
                lookup_kls = AbstractSet

        try:
            fn = self._unstructure_fns[lookup_kls]
        except KeyError:
            supported_expr = ", ".join(str(t) for t in self._structure_fns.keys())

            raise ValueError(
                f"`type_hint` must be of one of the following expressions, but is `{type_hint}` instead: {supported_expr}"
            ) from None

        try:
            return fn(kls, kls_args, obj)
        except (TypeError, ValueError) as ex:
            raise TypeError(
                f"`obj` cannot be unstructured to `{type_hint}`. See nested exception for details."
            ) from ex

    @classmethod
    def _unstructure_identity(
        cls, kls: Any, kls_args: Tuple[Any, ...], obj: Any
    ) -> Any:
        cls._check_type(obj, kls)

        return obj

    def _unstructure_dataclass(
        self, kls: Any, kls_args: Tuple[Any, ...], obj: Any
    ) -> Any:
        if kls is DataClass:
            if not is_dataclass_instance(obj):
                raise TypeError("`obj` must be a dataclass.")
        else:
            self._check_type(obj, kls)

        output = {}

        obj_kls = type(obj)

        if obj_kls is not kls:
            obj_kls_name = self._type_to_str(obj_kls)

            output["_type_"] = obj_kls_name

            kls = obj_kls

        type_hints = get_type_hints(kls)

        for field in fields(kls):
            value = getattr(obj, field.name)
            if value is EMPTY:
                output[field.name] = _EMPTY_TAG

                continue

            try:
                output[field.name] = self.unstructure(value, type_hints[field.name])
            except TypeError as ex:
                raise TypeError(
                    f"'{field.name}' field cannot be unstructured to `{type_hints[field.name]}`. See nested exception for details."
                ) from ex

        return output

    @classmethod
    def _unstructure_dtype(cls, kls: Any, kls_args: Tuple[Any, ...], obj: Any) -> Any:
        cls._check_type(obj, DataType)

        return str(obj)[6:]  # strip 'torch.'

    @classmethod
    def _unstructure_device(cls, kls: Any, kls_args: Tuple[Any, ...], obj: Any) -> Any:
        cls._check_type(obj, Device)

        return str(obj)

    @classmethod
    def _unstructure_enum(cls, kls: Any, kls_args: Tuple[Any, ...], obj: Any) -> Any:
        cls._check_type(obj, kls)

        return obj.name

    @classmethod
    def _unstructure_literal(cls, kls: Any, kls_args: Tuple[Any, ...], obj: Any) -> Any:
        cls._check_type(obj, str)

        if obj not in kls_args:
            values = ", ".join(kls_args)

            raise ValueError(
                f"`obj` must be one of the following values, but is '{obj}' instead: {values}"
            )

        return obj

    def _unstructure_mapping(
        self, kls: Any, kls_args: Tuple[Any, ...], obj: Any
    ) -> Any:
        self._check_type(obj, Mapping)

        if len(kls_args) != 2:
            raise TypeError(f"`type_hint` has no type annotation for `{kls}`.")

        output = {}

        for k, v in obj.items():
            k = self.unstructure(k, kls_args[0])
            v = self.unstructure(v, kls_args[1])

            output[k] = v

        return output

    @classmethod
    def _unstructure_path(cls, kls: Any, kls_args: Tuple[Any, ...], obj: Any) -> Any:
        cls._check_type(obj, Path)

        return str(obj)

    def _unstructure_list(self, kls: Any, kls_args: Tuple[Any, ...], obj: Any) -> Any:
        self._check_type(obj, Sequence)

        if len(kls_args) != 1:
            raise TypeError(f"`type_hint` has no type annotation for `{kls}`.")

        return [self.unstructure(e, kls_args[0]) for e in obj]

    def _unstructure_set(self, kls: Any, kls_args: Tuple[Any, ...], obj: Any) -> Any:
        self._check_type(obj, AbstractSet)

        if len(kls_args) != 1:
            raise TypeError(f"`type_hint` has no type annotation for `{kls}`.")

        return [self.unstructure(e, kls_args[0]) for e in obj]

    def _unstructure_tuple(self, kls: Any, kls_args: Tuple[Any, ...], obj: Any) -> Any:
        self._check_type(obj, tuple)

        num_args = len(kls_args)

        if num_args == 0:
            raise TypeError(f"`type_hint` has no type annotation for `{kls}`.")

        if num_args == 2 and kls_args[1] is Ellipsis:  # homogeneous
            return [self.unstructure(e, kls_args[0]) for e in obj]

        output = []

        for i, e in enumerate(obj):
            output.append(self.structure(e, kls_args[i]))

        return output

    def _unstructure_union(self, kls: Any, kls_args: Tuple[Any, ...], obj: Any) -> Any:
        is_optional = len(kls_args) == 2 and NoneType in kls_args

        if is_optional and obj is None:
            return obj

        for kls_ in kls_args:
            try:
                return self.unstructure(obj, kls_)
            except (TypeError, ValueError):
                if is_optional:
                    raise

                continue

        types = ", ".join(str(t) for t in kls_args)

        raise TypeError(
            f"`obj` must be parseable as one of the following union types: {types}"
        )

    @staticmethod
    def _str_to_type(s: str) -> type:
        names = s.rsplit(".", 1)

        if len(names) == 1:
            kls_name, module = names[0], builtins
        else:
            kls_name, module = names[1], import_module(names[0])

        kls = getattr(module, kls_name)

        if not isinstance(kls, type):
            raise ValueError(f"`{kls}` must be a type.")

        return kls  # type: ignore[no-any-return]

    @staticmethod
    def _type_to_str(kls: type) -> str:
        kls_name, module_name = kls.__name__, kls.__module__

        if module_name == "builtins":
            return kls_name

        return f"{module_name}.{kls_name}"

    @staticmethod
    def _check_type(obj: Any, kls: type) -> None:
        if not isinstance(obj, kls):
            raise TypeError(
                f"`obj` must be of type `{kls}`, but is of type `{type(obj)}` instead."
            )


default_value_converter = ValueConverter()
