# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Protocol, final, runtime_checkable

from typing_extensions import override

from fairseq2.typing import is_dataclass_instance


class ObjectValidator(ABC):
    @abstractmethod
    def validate(self, obj: object) -> None: ...


@final
class StandardObjectValidator(ObjectValidator):
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

        return result


@runtime_checkable
class Validatable(Protocol):
    def validate(self) -> ValidationResult: ...


@final
class ValidationResult:
    def __init__(self) -> None:
        self._errors: list[str] = []
        self._sub_results: dict[str, ValidationResult] = {}

    def add_error(self, message: str) -> None:
        self._errors.append(message)

    def add_sub_result(self, field: str, result: ValidationResult) -> None:
        self._sub_results[field] = result

    def has_error(self) -> bool:
        if self._errors:
            return True

        for sub_result in self._sub_results.values():
            if sub_result.has_error():
                return True

        return False

    @property
    def errors(self) -> Sequence[str]:
        return self._errors

    @property
    def sub_results(self) -> Mapping[str, ValidationResult]:
        return self._sub_results

    def __str__(self) -> str:
        parts: list[str] = []

        self._create_error_str(parts, path=[])

        return " ".join(parts)

    def _create_error_str(self, parts: list[str], path: list[str]) -> None:
        s = " ".join(self._errors)
        if s:
            if path:
                pathname = ".".join(path)

                parts.append(f"`{pathname}` is not valid: {s}")
            else:
                parts.append(s)

        for field, result in self._sub_results.items():
            path.append(field)

            result._create_error_str(parts, path)

            path.pop()


class ValidationError(Exception):
    def __init__(
        self, result: ValidationResult | str, *, field: str | None = None
    ) -> None:
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
