# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import pytest

from fairseq2.data.text.converters import StrToIntConverter


class TestStrToIntConverter:
    @pytest.mark.parametrize("value", [4, 1000, -34, 56])
    def test_call_works(self, value: int) -> None:
        converter = StrToIntConverter()

        assert converter(str(value)) == value

    def test_call_works_when_base_is_specified(self) -> None:
        converter = StrToIntConverter(base=8)

        assert converter("034") == 0o34

    @pytest.mark.parametrize("value", ["", "foo", "12foo", "foo12", "12foo12"])
    def test_call_raises_error_when_input_is_invalid(self, value: str) -> None:
        converter = StrToIntConverter()

        with pytest.raises(
            ValueError,
            match=rf"^The input string must represent a 64-bit integer, but is '{value}' instead\.$",
        ):
            converter(value)

    def test_call_raises_error_when_input_is_out_of_range(self) -> None:
        value = "999999999999999999999999999999999999"

        converter = StrToIntConverter()

        with pytest.raises(
            ValueError,
            match=rf"^The input string must represent a 64-bit integer, but is '{value}' instead, which is out of range\.$",
        ):
            converter(value)

    @pytest.mark.parametrize(
        "value,type_name", [(None, "pyobj"), (123, "int"), (1.2, "float")]
    )
    def test_raises_error_when_input_is_not_string(
        self, value: Any, type_name: str
    ) -> None:
        converter = StrToIntConverter()

        with pytest.raises(
            ValueError,
            match=rf"^The input data must be of type `string`, but is of type `{type_name}` instead\.$",
        ):
            converter(value)
