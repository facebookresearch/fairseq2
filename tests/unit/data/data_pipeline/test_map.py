# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import re

import pytest

from fairseq2.data import read_sequence
from fairseq2.data.text.converters import StrToIntConverter


class TestMapOp:
    @pytest.mark.parametrize("num_parallel_calls", [0, 1, 4, 10, 20])
    def test_op_works_as_expected(self, num_parallel_calls: int) -> None:
        def fn(d: int) -> int:
            return d**2

        seq = list(range(1, 10))

        dp = read_sequence(seq).map(fn, None, num_parallel_calls).and_return()

        for _ in range(2):
            assert list(dp) == [i**2 for i in seq]

            dp.reset()

    def test_op_works_with_data_processor_as_expected(self) -> None:
        fn = StrToIntConverter()

        dp = read_sequence(["1", "2", "3", "4"]).map(fn).and_return()

        for _ in range(2):
            assert list(dp) == [1, 2, 3, 4]

            dp.reset()

    def test_op_works_with_function_array_as_expected(self) -> None:
        fn1 = StrToIntConverter()

        def fn2(d: int) -> int:
            return d**2

        dp = read_sequence(["1", "2", "3", "4"]).map([fn1, fn2]).and_return()

        for _ in range(2):
            assert list(dp) == [1, 4, 9, 16]

            dp.reset()

    @pytest.mark.parametrize("num_parallel_calls", [0, 1, 4, 10, 20])
    def test_op_works_with_warn_only_as_expected(self, num_parallel_calls: int) -> None:
        def fn(d: int) -> int:
            if d % 2 == 0:
                raise ValueError("foo")

            return d**2

        seq = list(range(1, 10))

        dp = read_sequence(seq).map(fn, None, num_parallel_calls, True).and_return()

        for _ in range(2):
            assert list(dp) == [i**2 for i in range(1, 10, 2)]

            dp.reset()

    def test_op_works_with_basic_selector_as_expected(self) -> None:
        def fn1(d: int) -> int:
            return d + 10

        def fn2(d: int) -> int:
            return d * 2

        seq = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        dp = read_sequence(seq).map([fn1, fn2], selector="[1]").and_return()

        for _ in range(2):
            it = iter(dp)

            assert next(it) == [1, 24, 3]
            assert next(it) == [4, 30, 6]
            assert next(it) == [7, 36, 9]

            dp.reset()

    def test_op_works_with_complex_selector_as_expected(self) -> None:
        def fn(d: int) -> int:
            return d + 10

        d1 = {
            "foo1": 1,
            "foo2": [2, 3, {"foo4": 4}],
            "foo3": [5],
        }
        d2 = {
            "foo1": 6,
            "foo2": [7, 8, {"foo4": 9}],
            "foo3": [0],
        }

        e1 = copy.deepcopy(d1)
        e2 = copy.deepcopy(d2)

        e1["foo2"][2]["foo4"] = 14  # type: ignore[index]
        e2["foo2"][2]["foo4"] = 19  # type: ignore[index]

        dp = read_sequence([d1, d2]).map(fn, selector="foo2[2].foo4").and_return()

        for _ in range(2):
            it = iter(dp)

            assert next(it) == e1
            assert next(it) == e2

            dp.reset()

    def test_op_works_with_multiple_selectors_as_expected(self) -> None:
        def fn(d: int) -> int:
            return d + 10

        d1 = {
            "foo1": 1,
            "foo2": [2, 3, {"foo4": 4}],
            "foo3": [5],
        }
        d2 = {
            "foo1": 6,
            "foo2": [7, 8, {"foo4": 9}],
            "foo3": [0],
        }

        e1 = copy.deepcopy(d1)
        e2 = copy.deepcopy(d2)

        e1["foo1"] = 11
        e2["foo1"] = 16
        e1["foo2"][2]["foo4"] = 14  # type: ignore[index]
        e2["foo2"][2]["foo4"] = 19  # type: ignore[index]
        e1["foo3"] = [15]
        e2["foo3"] = [10]

        selector = "foo2[2].foo4,foo3[0], foo1"

        dp = read_sequence([d1, d2]).map(fn, selector=selector).and_return()

        for _ in range(2):
            it = iter(dp)

            assert next(it) == e1
            assert next(it) == e2

            dp.reset()

    @pytest.mark.parametrize(
        "s",
        [
            "[0]",
            "[0][1]",
            "foo",
            "  foo ",
            "foo1.foo2",
            "foo[0]",
            "foo[0][1]",
            "foo1.foo2[0]",
            "foo1,foo2",
            "foo1[0],foo2[0]",
            " foo1[0]  , foo2[1],foo3",
        ],
    )
    def test_op_accepts_well_formatted_selectors(self, s: str) -> None:
        read_sequence([]).map(lambda x: x, selector=s).and_return()

    @pytest.mark.parametrize(
        "s",
        [
            "",
            "  ",
            ".",
            "foo.",
            "foo[[0]",
            "foo[",
            "foo[]",
            "foo[0",
            "foo.[0]",
            ".foo",
            ",",
            " , ",
            "foo,",
            " , foo",
            "fo o",
            "foo [0]",
        ],
    )
    def test_op_raises_error_if_selector_is_not_well_formatted(self, s: str) -> None:
        with pytest.raises(
            ValueError,
            match=rf"^`selector` must contain one or more well-formatted element paths, but is '{re.escape(s)}' instead\.$",
        ):
            read_sequence([]).map(lambda x: x, selector=s).and_return()

    @pytest.mark.parametrize("s", ["[0]", "foo1.foo2", "foo2[3]", "foo1[1].foo3"])
    def test_op_raises_error_if_selector_is_not_valid(self, s: str) -> None:
        d = {
            "foo1": 1,
            "foo2": [2, 3, {"foo4": 4}],
            "foo3": [5],
        }

        dp = read_sequence([d]).map(lambda x: x, selector=s).and_return()

        with pytest.raises(
            ValueError,
            match=rf"^The input data does not have an element at path '{re.escape(s)}'\.$",
        ):
            next(iter(dp))

    @pytest.mark.parametrize("num_parallel_calls", [0, 1, 4, 20])
    def test_op_propagates_errors_as_expected(self, num_parallel_calls: int) -> None:
        def fn(d: int) -> int:
            if d == 3:
                raise ValueError("map error")

            return d

        dp = read_sequence([1, 2, 3, 4]).map(fn, None, num_parallel_calls).and_return()

        with pytest.raises(ValueError, match=r"^map error$"):
            for d in dp:
                pass

    @pytest.mark.parametrize("num_parallel_calls", [0, 1, 4, 20])
    def test_record_reload_position_works_as_expected(
        self, num_parallel_calls: int
    ) -> None:
        def fn(d: int) -> int:
            return d

        seq = list(range(1, 10))

        dp = read_sequence(seq).map(fn, None, num_parallel_calls).and_return()

        d = None

        it = iter(dp)

        # Move the the second example.
        for _ in range(2):
            d = next(it)

        assert d == 2

        state_dict = dp.state_dict()

        # Read a few examples before we roll back.
        for _ in range(4):
            d = next(it)

        assert d == 6

        # Expected to roll back to the second example.
        dp.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(7):
            d = next(it)

        assert d == 9

        state_dict = dp.state_dict()

        dp.reset()

        # Expected to be EOD.
        dp.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(dp))
