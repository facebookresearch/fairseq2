# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import os
import re
import time
from dataclasses import dataclass

import pytest

from fairseq2.data.data_pipeline import read_sequence
from fairseq2.data.text import StrToIntConverter


class TestMapOp:
    @pytest.mark.parametrize("num_parallel_calls", [1, 4, 10, 20])
    @pytest.mark.parametrize("deterministic", [True, False])
    def test_op_works(self, num_parallel_calls: int, deterministic: bool) -> None:
        def fn(d: int) -> int:
            return d**2

        seq = list(range(1, 10))

        pipeline = (
            read_sequence(seq)
            .map(fn, num_parallel_calls=num_parallel_calls, deterministic=deterministic)
            .and_return()
        )

        for _ in range(2):
            if deterministic:
                assert list(pipeline) == [i**2 for i in seq]
            else:
                assert set(pipeline) == {i**2 for i in seq}

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

    def test_op_works_when_selector_has_wildcard(self) -> None:
        def fn(d: int) -> int:
            return d + 10

        d = [
            {"foo1": [{"foo2": 2, "foo3": [1, 2]}]},
            {"foo1": [{"foo2": 2, "foo3": [3, 4]}]},
            {"foo1": [{"foo2": 2, "foo3": [5, 6]}]},
        ]

        e = copy.deepcopy(d)

        e[0]["foo1"][0]["foo3"] = [11, 12]
        e[1]["foo1"][0]["foo3"] = [13, 14]
        e[2]["foo1"][0]["foo3"] = [15, 16]

        selector = "[*].foo1[*].foo3[*]"

        pipeline = read_sequence([d]).map(fn, selector=selector).and_return()

        for _ in range(2):
            it = iter(pipeline)

            assert next(it) == e

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
            "foo[*]",
            "foo[*][2]",
            "foo[1][*]",
            "foo1.foo2[0]",
            "foo1,foo2",
            "foo1[0],foo2[0]",
            " foo1[0]  , foo2[1],foo3,  foo[*][3]",
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
            "foo[*",
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

        with pytest.raises(ValueError) as exc_info:
            next(iter(pipeline))

        assert str(exc_info.value) == f"The input data does not have an element at path '{s}'."  # fmt: skip

    def test_op_raises_error_when_selector_with_wildcard_is_not_valid(self) -> None:
        d = [
            {"foo1": 0},
            {"foo2": 1},
            {"foo1": 2},
        ]

        s = "[*].foo1"

        pipeline = read_sequence([d]).map(lambda x: x, selector=s).and_return()

        with pytest.raises(ValueError) as exc_info:
            next(iter(pipeline))

        assert str(exc_info.value) == "The input data does not have an element at path '[1].foo1'."  # fmt: skip

    @pytest.mark.parametrize("num_parallel_calls", [1, 4, 20])
    def test_op_raises_error_when_callable_fails(self, num_parallel_calls: int) -> None:
        def fn(d: int) -> int:
            if d == 3:
                raise ValueError("map error")

            return d

        pipeline = read_sequence([1, 2, 3, 4]).map(fn, num_parallel_calls=num_parallel_calls).and_return()  # fmt: skip

        with pytest.raises(ValueError) as exc_info:
            for d in pipeline:
                pass

        assert str(exc_info.value) == "map error"

    @pytest.mark.parametrize("num_parallel_calls", [1, 4, 20])
    def test_op_saves_and_restores_its_state(self, num_parallel_calls: int) -> None:
        def fn(d: int) -> int:
            return d

        seq = list(range(1, 10))

        pipeline = read_sequence(seq).map(fn, num_parallel_calls=num_parallel_calls).and_return()  # fmt: skip

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

    @pytest.mark.parametrize("num_parallel_calls", [1, 4, 20])
    def test_op_saves_and_restores_its_state_non_deterministic(self, num_parallel_calls: int) -> None:  # fmt: skip

        def fn(d: int) -> int:
            time.sleep(0.05 if d == 5 else 0.0)
            return d

        seq = list(range(1, 10))

        pipeline = (
            read_sequence(seq)
            .map(fn, num_parallel_calls=num_parallel_calls, deterministic=False)
            .and_return()
        )

        remaining = set(seq)
        seen = set()

        d = None

        it = iter(pipeline)

        # Move to the second example.
        for _ in range(2):
            d = next(it)
            assert d in remaining
            remaining.remove(d)
            seen.add(d)

        state_dict = pipeline.state_dict()

        # Read a few examples before we roll back.
        for _ in range(4):
            d = next(it)

        # Expected to roll back to the second example.
        pipeline.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(7):
            d = next(it)
            assert d in remaining
            remaining.remove(d)
            seen.add(d)

        assert not remaining
        assert seen == set(seq)

        state_dict = pipeline.state_dict()

        pipeline.reset()

        # Expected to be EOD.
        pipeline.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            d = next(iter(pipeline))

    @pytest.mark.skipif(
        not hasattr(os, "sched_getaffinity") or len(os.sched_getaffinity(0)) < 2,
        reason="Not enough CPU cores available",
    )
    @pytest.mark.parametrize("num_parallel_calls", [1, 4, 20])
    @pytest.mark.parametrize("nb_elements", [10, 20])
    def test_return_order_non_deterministic(self, num_parallel_calls: int, nb_elements: int) -> None:  # fmt: skip

        def fn(d: int) -> int:
            time.sleep(0.2 if d == 5 else 0.0)
            return d

        seq = list(range(nb_elements))
        pipeline = (
            read_sequence(seq)
            .map(fn, num_parallel_calls=num_parallel_calls, deterministic=False)
            .and_return()
        )

        return_seq = list(pipeline)
        assert len(return_seq) == nb_elements
        assert seq == sorted(return_seq)
        if num_parallel_calls == 1:
            assert return_seq == seq
        else:
            assert return_seq[-1] == 5  # 5 is the slowest one
