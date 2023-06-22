# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import pytest

from fairseq2.data.processors import StrToIntConverter


class TestStrToIntConverter:
    @pytest.mark.parametrize("value", [4, 1000, -34, 56])
    def test_converts_as_expected(self, value: int) -> None:
        converter = StrToIntConverter()

        i = converter(str(value))

        assert i == value

    def test_converts_non_default_base_as_expected(self) -> None:
        converter = StrToIntConverter(base=8)

        i = converter("034")

        assert i == 0o34

    @pytest.mark.parametrize("value", ["", "foo", "12foo", "foo12", "12foo12"])
    def test_raises_error_if_string_is_invalid(self, value: str) -> None:
        converter = StrToIntConverter()

        with pytest.raises(
            ValueError,
            match=rf"^The input string must be an integer, but is '{value}' instead\.$",
        ):
            converter(value)

    def test_raises_error_if_string_is_out_of_range(self) -> None:
        value = "999999999999999999999999999999999999"

        converter = StrToIntConverter()

        with pytest.raises(
            ValueError,
            match=rf"^The input string must be a signed 64-bit integer, but is '{value}' instead\.$",
        ):
            converter(value)

    @pytest.mark.parametrize("value", [None, 123, 1.2])
    def test_raises_error_if_input_is_not_string(self, value: Any) -> None:
        converter = StrToIntConverter()

        with pytest.raises(
            ValueError, match=r"^The input data must be of type string\.$"
        ):
            converter(value)
