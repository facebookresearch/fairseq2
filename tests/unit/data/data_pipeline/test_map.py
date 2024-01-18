# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import re
from dataclasses import dataclass

import pytest

from fairseq2.data import DataPipelineError, read_sequence
from fairseq2.data.text.converters import StrToIntConverter


class TestMapOp:
    @pytest.mark.parametrize("num_parallel_calls", [0, 1, 4, 10, 20])
    def test_op_works(self, num_parallel_calls: int) -> None:
        def fn(d: int) -> int:
            return d**2

        seq = list(range(1, 10))

        pipeline = (
            read_sequence(seq)
            .map(fn, num_parallel_calls=num_parallel_calls)
            .and_return()
        )

        for _ in range(2):
            assert list(pipeline) == [i**2 for i in seq]

            pipeline.reset()

    def test_op_works_when_callable_is_native(self) -> None:
        fn = StrToIntConverter()

        pipeline = read_sequence(["1", "2", "3", "4"]).map(fn).and_return()

        for _ in range(2):
            assert list(pipeline) == [1, 2, 3, 4]

            pipeline.reset()

    def test_op_works_when_callable_is_a_list(self) -> None:
        fn1 = StrToIntConverter()

        def fn2(d: int) -> int:
            return d**2

        pipeline = read_sequence(["1", "2", "3", "4"]).map([fn1, fn2]).and_return()

        for _ in range(2):
            assert list(pipeline) == [1, 4, 9, 16]

            pipeline.reset()

    def test_op_works_when_input_is_python_object(self) -> None:
        @dataclass
        class Foo:
            value: int

        def fn(d: Foo) -> Foo:
            d.value += 2

            return d

        pipeline = read_sequence([Foo(1), Foo(2)]).map(fn).and_return()

        it = iter(pipeline)

        for i in range(1, 3):
            assert next(it) == Foo(1 + (i * 2))
            assert next(it) == Foo(2 + (i * 2))

            pipeline.reset()

    def test_op_works_when_selector_is_basic(self) -> None:
        def fn1(d: int) -> int:
            return d + 10

        def fn2(d: int) -> int:
            return d * 2

        seq = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        pipeline = read_sequence(seq).map([fn1, fn2], selector="[1]").and_return()

        for _ in range(2):
            it = iter(pipeline)

            assert next(it) == [1, 24, 3]
            assert next(it) == [4, 30, 6]
            assert next(it) == [7, 36, 9]

            pipeline.reset()

    def test_op_works_when_selector_is_complex(self) -> None:
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

        pipeline = read_sequence([d1, d2]).map(fn, selector="foo2[2].foo4").and_return()

        for _ in range(2):
            it = iter(pipeline)

            assert next(it) == e1
            assert next(it) == e2

            pipeline.reset()

    def test_op_works_when_selector_has_multiple_paths(self) -> None:
        def fn(d: int) -> int:
            return d + 10

        d1 = {
            "foo1": 1,
            "foo2": [2, 3, {"foo4": 4}],
            "foo3": [5],
            "foo5": {"foo6": {"foo7": 1}},
        }
        d2 = {
            "foo1": 6,
            "foo2": [7, 8, {"foo4": 9}],
            "foo3": [0],
            "foo5": {"foo6": {"foo7": 2}},
        }

        e1 = copy.deepcopy(d1)
        e2 = copy.deepcopy(d2)

        e1["foo1"] = 11
        e2["foo1"] = 16
        e1["foo2"][2]["foo4"] = 14  # type: ignore[index]
        e2["foo2"][2]["foo4"] = 19  # type: ignore[index]
        e1["foo3"] = [15]
        e2["foo3"] = [10]
        e1["foo5"]["foo6"]["foo7"] = 11  # type: ignore[index]
        e2["foo5"]["foo6"]["foo7"] = 12  # type: ignore[index]

        selector = "foo2[2].foo4,foo3[0], foo1,foo5.foo6.foo7"

        pipeline = read_sequence([d1, d2]).map(fn, selector=selector).and_return()

        for _ in range(2):
            it = iter(pipeline)

            assert next(it) == e1
            assert next(it) == e2

            pipeline.reset()

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
    def test_op_inits_when_selectors_are_well_formatted(self, s: str) -> None:
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
            "foo[999999999999999999999999999999]",  # overflow
        ],
    )
    def test_op_raises_error_when_selector_is_not_well_formatted(self, s: str) -> None:
        with pytest.raises(
            ValueError,
            match=rf"^`selector` must contain one or more well-formatted element paths, but is '{re.escape(s)}' instead\.$",
        ):
            read_sequence([]).map(lambda x: x, selector=s).and_return()

    @pytest.mark.parametrize("s", ["[0]", "foo1.foo2", "foo2[3]", "foo1[1].foo3"])
    def test_op_raises_error_when_selector_is_not_valid(self, s: str) -> None:
        d = {
            "foo1": 1,
            "foo2": [2, 3, {"foo4": 4}],
            "foo3": [5],
        }

        pipeline = read_sequence([d]).map(lambda x: x, selector=s).and_return()

        with pytest.raises(DataPipelineError) as exc_info:
            next(iter(pipeline))

        cause = exc_info.value.__cause__

        assert isinstance(cause, ValueError)

        assert str(cause) == f"The input data does not have an element at path '{s}'."

    @pytest.mark.parametrize("num_parallel_calls", [0, 1, 4, 20])
    def test_op_raises_nested_error_when_callable_fails(
        self, num_parallel_calls: int
    ) -> None:
        def fn(d: int) -> int:
            if d == 3:
                raise ValueError("map error")

            return d

        pipeline = (
            read_sequence([1, 2, 3, 4])
            .map(fn, num_parallel_calls=num_parallel_calls)
            .and_return()
        )

        with pytest.raises(DataPipelineError) as exc_info:
            for d in pipeline:
                pass

        cause = exc_info.value.__cause__

        assert isinstance(cause, ValueError)

        assert str(cause) == "map error"

    @pytest.mark.parametrize("num_parallel_calls", [0, 1, 4, 20])
    def test_op_saves_and_restores_its_state(self, num_parallel_calls: int) -> None:
        def fn(d: int) -> int:
            return d

        seq = list(range(1, 10))

        pipeline = (
            read_sequence(seq)
            .map(fn, num_parallel_calls=num_parallel_calls)
            .and_return()
        )

        d = None

        it = iter(pipeline)

        # Move to the second example.
        for _ in range(2):
            d = next(it)

        assert d == 2

        state_dict = pipeline.state_dict()

        # Read a few examples before we roll back.
        for _ in range(4):
            d = next(it)

        assert d == 6

        # Expected to roll back to the second example.
        pipeline.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(7):
            d = next(it)

        assert d == 9

        state_dict = pipeline.state_dict()

        pipeline.reset()

        # Expected to be EOD.
        pipeline.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(pipeline))
