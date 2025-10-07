# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This module provides a set of helper types for object state validation. These
types are primarly used by recipe and asset configuration dataclasses to ensure
that all fields are set correctly.

A class -*typically a configuration dataclass*- that wants to support validation
should expose a ``validate(self) -> ValidationResult`` method. Optionally,
the class can derive from the runtime-checkable :class:`Validatable` protocol
to make its intent more clear.

A typical implementation of a ``validate()`` method looks like the following:

.. code:: python

    from dataclasses import dataclass

    from fairseq2.utils.validation import Validatable, ValidationResult

    @dataclass
    class FooConfig(Validatable):
        field1: str
        field2: int

        def validate(self) -> ValidationResult:
            result = ValidationResult()

            if not self.field1:
                result.add_error("`field1` must be a non-empty string.")

            if self.field2 < 1:
                result.add_error("`field2` must be a positive integer.")

            return result


Note that ``FooConfig`` must NOT call ``validate()`` on its sub-fields that are
validatable. :class:`ObjectValidator` will traverse the object graph and call
each ``validate()`` method it finds in dataclasses as well as in composite
objects of types ``list``, ``Mapping``, ``Set``, and ``tuple``.

Whenever ``FooConfig`` is used in a recipe configuration, fairseq2 will ensure
that it is validated before setting :attr:`RecipeContext.config`. To manually
validate an object outside of recipes, :class:`StandardObjectValidator` can
be used:

.. code:: python

    from dataclasses import dataclass

    from fairseq2.utils.validation import (
        ObjectValidator,
        StandardObjectValidator,
        Validatable,
        ValidationError,
        ValidationResult,
    )

    @dataclass
    class FooConfig(Validatable):
        field1: str
        field2: int

        def validate(self) -> ValidationResult:
            result = ValidationResult()

            if not self.field1:
                result.add_error("`field1` must be a non-empty string.")

            if self.field2 < 1:
                result.add_error("`field2` must be a positive integer.")

            return result

    config = FooConfig(field1="foo", field2=0)

    validator: ObjectValidator = StandardObjectValidator()

    try:
        validator.validate(config)
    except ValidationError as ex:
        # Prints an error message indicating that `field2` must be a
        # positive integer.
        print(ex.result)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence, Set
from dataclasses import fields
from typing import Protocol, final, runtime_checkable

from typing_extensions import override

from fairseq2.typing import is_dataclass_instance


class ObjectValidator(ABC):
    """
    Validates an object along with its sub-objects if it is a composite object
    (i.e. a ``dataclass``, ``list``, ``Mapping``, ``Set``, or ``tuple``) and
    raises a :class:`ValidationError` if any of them returns an error.
    """

    @abstractmethod
    def validate(self, obj: object) -> None:
        """
        Validates ``obj``.

        :raises ValidationError: If ``obj`` or one of its sub-objects has a
            validation error.
        """


@final
class StandardObjectValidator(ObjectValidator):
    """Represents the standard implementation of :class:`ObjectValidator`."""

    @override
    def validate(self, obj: object) -> None:
        result = self._do_validate(obj)

        if result.has_error():
            raise ValidationError(result)

    def _do_validate(self, obj: object) -> ValidationResult:
        if isinstance(obj, Validatable):
            result = obj.validate()
        else:
            result = ValidationResult()

        if is_dataclass_instance(obj):
            for field in fields(obj):
                value = getattr(obj, field.name)

                sub_result = self._do_validate(value)

                if sub_result.has_error():
                    result.add_sub_result(field.name, sub_result)
        elif isinstance(obj, Mapping):
            for k, v in obj.items():
                sub_result = self._do_validate(v)

                if sub_result.has_error():
                    result.add_sub_result(f"[{repr(k)}]", sub_result)
        elif isinstance(obj, (list, tuple, Set)):
            for i, v in enumerate(obj):
                sub_result = self._do_validate(v)

                if sub_result.has_error():
                    result.add_sub_result(f"[{i}]", sub_result)

        return result


@runtime_checkable
class Validatable(Protocol):
    """Represents the protocol for validatable objects."""

    def validate(self) -> ValidationResult:
        """Validates the state of the object."""


@final
class ValidationResult:
    """Holds the result of a :meth:`~Validatable.validate` call."""

    def __init__(self) -> None:
        self._errors: list[str] = []
        self._sub_results: dict[str, ValidationResult] = {}

    def add_error(self, message: str) -> None:
        """Adds an error message to the result."""
        self._errors.append(message)

    def add_sub_result(self, field: str, result: ValidationResult) -> None:
        """
        Adds the validation result of a sub-object as a sub-result.

        ``field`` specifies the name of the sub-object. For a dataclass, it is
        the name of the field, for a ``Mapping``, it is the name of the key
        formatted as ``f"[{repr(key)}]"``, for a ``list``, ``Set``, or ``tuple``,
        it is the index of the value formatted as ``f"[{index}]"``.
        """
        self._sub_results[field] = result

    def has_error(self) -> bool:
        """
        Returns ``True`` if the object or any of its sub-objects have a
        validation error.
        """
        if self._errors:
            return True

        return any(r.has_error() for r in self._sub_results.values())

    @property
    def errors(self) -> Sequence[str]:
        """
        Returns the validation errors of the object, excluding errors of its
        sub-objects.
        """
        return self._errors

    @property
    def sub_results(self) -> Mapping[str, ValidationResult]:
        """Returns the validation results of the sub-objects."""
        return self._sub_results

    def __str__(self) -> str:
        output: list[str] = []

        self._create_error_string(output, field_path=[])

        return " ".join(output)

    def _create_error_string(self, output: list[str], field_path: list[str]) -> None:
        s = " ".join(self._errors)
        if s:
            if field_path:
                pathname = self._build_pathname(field_path)

                output.append(f"`{pathname}` is not valid: {s}")
            else:
                output.append(s)

        for field, result in self._sub_results.items():
            field_path.append(field)

            result._create_error_string(output, field_path)

            field_path.pop()

    def _build_pathname(self, field_path: list[str]) -> str:
        segments = [field_path[0]]

        for p in field_path[1:]:
            if not p.startswith("[") or not p.endswith("]"):
                segments.append(".")

            segments.append(p)

        return "".join(segments)


class ValidationError(Exception):
    """Raised when a validation error occurs."""

    result: ValidationResult

    def __init__(
        self, result: ValidationResult | str, *, field: str | None = None
    ) -> None:
        """
        If ``field`` is provided, ``result`` will be treated as the sub-result
        of the specified field.
        """
        if isinstance(result, str):
            tmp = ValidationResult()

            tmp.add_error(result)

            result = tmp

        if field is not None:
            tmp = ValidationResult()

            tmp.add_sub_result(field, result)

            result = tmp

        self.result = result

    def __str__(self) -> str:
        return str(self.result)
