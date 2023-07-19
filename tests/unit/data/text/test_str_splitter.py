# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

import pytest

from fairseq2.data.text.converters import StrSplitter


class TestStrSplitter:
    def test_init_raises_error_when_names_and_indices_have_different_lengths(
        self,
    ) -> None:
        with pytest.raises(
            ValueError,
            match=r"^`names` and `indices` must have the same length, but have the lengths 2 and 3 instead\.$",
        ):
            StrSplitter(names=["a", "b"], indices=[1, 2, 3])

    def test_init_does_not_raise_error_when_names_and_exclude_indices_have_different_lengths(
        self,
    ) -> None:
        StrSplitter(names=["a", "b"], indices=[1, 2, 3], exclude=True)

    def test_call_works(self) -> None:
        # fmt: off
        s = "23\t9\t12\t\tabc\t34\t~~\t\t90\t 1 \t "

        splitter = StrSplitter()

        assert splitter(s) == ["23", "9", "12", "", "abc", "34", "~~", "", "90", " 1 ", " "]
        # fmt: on

    def test_call_works_when_separator_is_specified(self) -> None:
        # fmt: off
        s = "23 9 12  abc 34 ~~  90 \t 1  "

        splitter = StrSplitter(sep=" ")

        assert splitter(s) == ["23", "9", "12", "", "abc", "34", "~~", "", "90", "\t", "1", "", ""]
        # fmt: on

    def test_call_works_when_input_is_empty(self) -> None:
        splitter = StrSplitter()

        assert splitter("") == [""]

    def test_call_works_when_names_are_specified(self) -> None:
        s = "1\t2\t3"

        splitter = StrSplitter(names=["a", "b", "c"])

        assert splitter(s) == {"a": "1", "b": "2", "c": "3"}

    @pytest.mark.parametrize("indices", [[0], [1], [4], [2, 3], [1, 2, 4]])
    def test_call_works_when_indices_are_specified(
        self, indices: Sequence[int]
    ) -> None:
        s = "0,1,2,3,4"

        splitter = StrSplitter(sep=",", indices=indices)

        assert splitter(s) == [str(i) for i in indices]

    @pytest.mark.parametrize(
        "indices,expected",
        [
            ([0], [1, 2, 3, 4]),
            ([4], [0, 1, 2, 3]),
            ([2, 3], [0, 1, 4]),
            ([1, 2, 4], [0, 3]),
        ],
    )
    def test_call_works_when_exclude_indices_are_specified(
        self, indices: Sequence[int], expected: Sequence[int]
    ) -> None:
        s = "0,1,2,3,4"

        splitter = StrSplitter(sep=",", indices=indices, exclude=True)

        assert splitter(s) == [str(i) for i in expected]

    @pytest.mark.parametrize("exclude", [False, True])
    def test_call_works_when_indices_is_empty(self, exclude: bool) -> None:
        s = "0,1,2,3,4"

        splitter = StrSplitter(sep=",", indices=[], exclude=exclude)

        assert splitter(s) == ["0", "1", "2", "3", "4"]

    @pytest.mark.parametrize("exclude", [False, True])
    def test_call_raises_error_when_the_number_of_fields_is_less_than_or_equal_to_highest_index(
        self, exclude: bool
    ) -> None:
        s = "0,1,2,3,4"

        splitter = StrSplitter(sep=",", indices=[1, 5, 7], exclude=exclude)

        with pytest.raises(
            ValueError,
            match=r"^The input string must have at least 7 field\(s\), but has 5 instead\.$",
        ):
            splitter(s)

    def test_call_raises_error_when_the_number_of_fields_and_names_do_not_match(
        self,
    ) -> None:
        s = "1\t2\t3"

        splitter = StrSplitter(names=["a", "b"])

        with pytest.raises(
            ValueError,
            match=r"^The number of fields must match the number of names \(2\), but is 3 instead\.$",
        ):
            splitter(s)
