# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from fairseq2.utils.validation import (
    StandardObjectValidator,
    Validatable,
    ValidationError,
    ValidationResult,
)


@dataclass
class FooConfig:
    field0: str
    field1: int
    field2: dict[str, FooSubConfig]
    field3: tuple[FooSubConfig, FooSubConfig]
    field4: FooComplexSubConfig

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if self.field0 == "invalid":
            result.add_error("root is not valid.")

        return result


@dataclass
class FooSubConfig(Validatable):
    def __init__(self, error: str | None = None) -> None:
        self.error = error

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if self.error is not None:
            result.add_error(self.error)

        return result


@dataclass
class FooComplexSubConfig:
    field4_1: int
    field4_2: list[FooSubConfig]
    field4_3: str


class TestStandardObjectValidator:
    def test_validate_works(self) -> None:
        config = FooConfig(
            field0="invalid",
            field1=2,
            field2={
                "foo2_1": FooSubConfig(),
                "foo2_2": FooSubConfig("value2_2 error."),
            },
            field3=(FooSubConfig("value3_1 error."), FooSubConfig()),
            field4=FooComplexSubConfig(
                field4_1=3, field4_2=[FooSubConfig("value4_2 error.")], field4_3="xyz"
            ),
        )

        validator = StandardObjectValidator()

        with pytest.raises(
            ValidationError, match=r"^root is not valid\. `field2\['foo2_2'\]` is not valid: value2_2 error\. `field3\[0\]` is not valid: value3_1 error\. `field4.field4_2\[0\]` is not valid: value4_2 error\.$"  # fmt: skip
        ):
            validator.validate(config)

    def test_validate_works_when_root_has_no_error(self) -> None:
        config = FooConfig(
            field0="abc",
            field1=2,
            field2={
                "foo2_1": FooSubConfig(),
                "foo2_2": FooSubConfig("value2_2 error."),
            },
            field3=(FooSubConfig(), FooSubConfig()),
            field4=FooComplexSubConfig(
                field4_1=3, field4_2=[FooSubConfig("value4_2 error.")], field4_3="xyz"
            ),
        )

        validator = StandardObjectValidator()

        with pytest.raises(
            ValidationError, match=r"^`field2\['foo2_2'\]` is not valid: value2_2 error\. `field4.field4_2\[0\]` is not valid: value4_2 error\.$"  # fmt: skip
        ):
            validator.validate(config)

    def test_validate_works_when_no_error(self) -> None:
        config = FooConfig(
            field0="abc",
            field1=2,
            field2={"foo2_1": FooSubConfig(), "foo2_2": FooSubConfig()},
            field3=(FooSubConfig(), FooSubConfig()),
            field4=FooComplexSubConfig(
                field4_1=3, field4_2=[FooSubConfig()], field4_3="xyz"
            ),
        )

        validator = StandardObjectValidator()

        validator.validate(config)


class TestValidationError:
    def test_init_works_with_str_result(self) -> None:
        error = ValidationError("value error")

        assert len(error.result.errors) == 1

        assert len(error.result.sub_results) == 0

        assert error.result.errors[0] == "value error"

    def test_init_works_with_field(self) -> None:
        error = ValidationError("value error", field="foo")

        assert len(error.result.errors) == 0

        assert len(error.result.sub_results) == 1

        assert "foo" in error.result.sub_results

        assert len(error.result.sub_results["foo"].errors) == 1

        assert len(error.result.sub_results["foo"].sub_results) == 0

        assert error.result.sub_results["foo"].errors[0] == "value error"
