# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from typing import Protocol, final, runtime_checkable

from fairseq2.typing import is_dataclass_instance


def validate(obj: object) -> None:
    if is_dataclass_instance(obj):
        for field in fields(obj):
            value = getattr(obj, field.name)

            validate(value)

    if isinstance(obj, Validatable):
        obj.validate()


@runtime_checkable
class Validatable(Protocol):
    def validate(self) -> None: ...


@final
class ValidationResult:
    _errors: list[str]

    def __init__(self) -> None:
        self._errors = []

    def add_error(self, message: str) -> None:
        self._errors.append(message)

    @property
    def has_error(self) -> bool:
        return bool(self._errors)

    @property
    def errors(self) -> Sequence[str]:
        return self._errors


class ValidationError(Exception):
    result: ValidationResult

    def __init__(self, message: str, result: ValidationResult) -> None:
        if result.has_error:
            errors = " ".join(result.errors)

            message = f"{message} {errors}"

        super().__init__(message)

        self.result = result
