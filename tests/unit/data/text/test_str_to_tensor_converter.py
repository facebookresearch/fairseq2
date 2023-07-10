# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional, Sequence

import pytest
import torch

from fairseq2.data.text.converters import StrToTensorConverter
from fairseq2.typing import DataType
from tests.common import assert_equal


class TestStrToTensorConverter:
    def test_init_raises_error_when_data_type_is_not_supported(self) -> None:
        with pytest.raises(ValueError, match=r"^Only integral types are supported\.$"):
            StrToTensorConverter(dtype=torch.half)

    @pytest.mark.parametrize("dtype", [None, torch.int16, torch.int32, torch.int64])
    def test_call_works(self, dtype: Optional[DataType]) -> None:
        s = "23 9 12  34   90 1  "

        converter = StrToTensorConverter(dtype=dtype)

        tensor = converter(s)

        assert_equal(
            tensor, torch.tensor([23, 9, 12, 34, 90, 1], dtype=dtype or torch.int32)
        )

    def test_call_works_when_input_is_empty_string(self) -> None:
        converter = StrToTensorConverter(dtype=torch.int32)

        tensor = converter("")

        assert_equal(tensor, torch.empty((0,), dtype=torch.int32))

    @pytest.mark.parametrize("size", [[3, 2], (3, 2), torch.Size([3, 2])])
    def test_call_works_when_size_is_specified(self, size: Sequence[int]) -> None:
        s = "23 9 12  34   90 1  "

        converter = StrToTensorConverter(size, dtype=torch.int32)

        tensor = converter(s)

        assert_equal(
            tensor, torch.tensor([[23, 9], [12, 34], [90, 1]], dtype=torch.int32)
        )

    def test_call_works_when_size_is_specified_with_an_unspecified_dimension(
        self,
    ) -> None:
        s = "23 9 12  34   90 1  "

        converter = StrToTensorConverter(size=(-1, 2), dtype=torch.int32)

        tensor = converter(s)

        assert_equal(
            tensor, torch.tensor([[23, 9], [12, 34], [90, 1]], dtype=torch.int32)
        )

    @pytest.mark.parametrize(
        "value,type_name", [(None, "pyobj"), (123, "int"), (1.2, "float")]
    )
    def test_call_raises_error_when_input_is_not_string(
        self, value: Any, type_name: str
    ) -> None:
        converter = StrToTensorConverter()

        with pytest.raises(
            ValueError,
            match=rf"^The input data must be of type `string`, but is of type `{type_name}` instead\.$",
        ):
            converter(value)

    @pytest.mark.parametrize("value", ["df", "12c", "b34", "23a54"])
    def test_call_raises_error_when_input_is_invalid(self, value: str) -> None:
        s = f"28 {value} 4 89"

        converter = StrToTensorConverter(dtype=torch.int32)

        with pytest.raises(
            ValueError,
            match=rf"^The input string must be a space-separated list representing values of type `torch.int32`, but contains an element with value '{value}' that cannot be parsed as `torch.int32`\.$",
        ):
            converter(s)

    def test_call_raises_error_when_input_is_out_of_range(self) -> None:
        value = "99999999999999999999"

        s = f"45 78 {value} 56"

        converter = StrToTensorConverter(dtype=torch.int16)

        with pytest.raises(
            ValueError,
            match=rf"^The input string must be a space-separated list representing values of type `torch.int16`, but contains an element with value '{value}' that is out of range for `torch.int16`\.$",
        ):
            converter(s)

    def test_call_raises_error_when_size_is_invalid(self) -> None:
        s = "23 9 12  34   90 1  "

        converter = StrToTensorConverter((5, 2), dtype=torch.int32)

        with pytest.raises(RuntimeError, match=r"^shape '\[5, 2\]' is invalid"):
            converter(s)
