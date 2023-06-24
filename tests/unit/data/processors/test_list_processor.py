# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence

import pytest
import torch

from fairseq2.data.processors import (
    ListProcessor,
    StrToIntConverter,
    StrToTensorConverter,
)
from tests.common import assert_equal


class TestListProcessor:
    @pytest.mark.parametrize("indices", [None, []])
    def test_processes_as_expected(self, indices: Optional[Sequence[int]]) -> None:
        l = ["45", "12 23 34 45"]

        processor = ListProcessor(
            StrToIntConverter(),
            StrToTensorConverter(dtype=torch.int16),
            indices=indices,
        )

        v = processor(l)

        assert len(v) == len(l)

        assert v[0] == 45

        assert_equal(v[1], torch.tensor([12, 23, 34, 45], dtype=torch.int16))

    def test_processes_with_indices_as_expected(self) -> None:
        l = ["ab", "45", "cd", "12 23 34 45", "ef"]

        processor = ListProcessor(
            StrToIntConverter(),
            StrToTensorConverter(dtype=torch.int16),
            indices=[1, 3],
        )

        v = processor(l)

        assert len(v) == len(l)

        assert v[0] == "ab"
        assert v[1] == 45
        assert v[2] == "cd"

        assert_equal(v[3], torch.tensor([12, 23, 34, 45], dtype=torch.int16))

    def test_processes_with_callables_as_expected(self) -> None:
        def fn(i: int) -> int:
            return i**2

        l = ["2", 3, "1 2 3 4"]

        processor = ListProcessor(int, fn, torch.tensor)

        v = processor(l)

        assert len(v) == len(l)

        assert v[0] == 2
        assert v[1] == 9

        assert_equal(v[2], torch.tensor([1, 2, 3, 4], dtype=torch.int32))

    def test_raises_error_if_the_number_of_processors_and_indices_do_not_match(
        self,
    ) -> None:
        with pytest.raises(
            ValueError,
            match=r"^`processors` and `indices` must have the same length, but have the lengths 2 and 3 instead\.$",
        ):
            ListProcessor(int, int, indices=[1, 2, 3])

    @pytest.mark.parametrize("indices", [[2, 1, 3], [1, 3, 2], [1, 1, 3], [1, 3, 3]])
    def test_raises_error_if_indices_are_not_sorted(
        self, indices: Optional[Sequence[int]]
    ) -> None:
        with pytest.raises(
            ValueError, match=r"^`indices` must be unique and in sorted order\.$"
        ):
            ListProcessor(int, int, int, indices=indices)

    def test_raises_error_if_the_length_of_input_is_invalid(self) -> None:
        l = ["1", "2", "3", "4"]

        processor = ListProcessor(int, int, int)

        with pytest.raises(
            ValueError,
            match=r"^The length of the input list must equal 3, but is 4 instead\.$",
        ):
            processor(l)

    def test_raises_error_if_the_length_of_input_is_invalid_with_indices(self) -> None:
        l = ["1", "2", "3", "4"]

        processor = ListProcessor(int, int, int, indices=[0, 4, 9])

        with pytest.raises(
            ValueError,
            match=r"^The length of the input list must be longer than 9, but is 4 instead\.$",
        ):
            processor(l)
