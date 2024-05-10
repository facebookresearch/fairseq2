# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import pytest

from fairseq2.utils.dataclass import FieldError, update_dataclass


@dataclass
class Foo2:
    x: Optional[int]
    y: str


@dataclass
class Foo1:
    a: int
    b: str
    c: Foo2
    d: Optional[Foo2]


class TestUpdateDataclassFunction:
    def test_call_works(self) -> None:
        obj = Foo1(a=1, b="b", c=Foo2(x=2, y="foo3"), d=Foo2(x=3, y="foo3"))

        overrides = {"b": "a", "c": {"x": None, "y": "foo4"}, "d": None}

        unknown_fields = update_dataclass(obj, overrides)

        assert obj == Foo1(a=1, b="a", c=Foo2(x=None, y="foo4"), d=None)

        assert unknown_fields == []

    def test_call_works_when_overrides_is_empty(self) -> None:
        obj = Foo1(a=1, b="b", c=Foo2(x=2, y="foo3"), d=Foo2(x=3, y="foo3"))

        update_dataclass(obj, {})

        assert obj == Foo1(a=1, b="b", c=Foo2(x=2, y="foo3"), d=Foo2(x=3, y="foo3"))

        unknown_fields = update_dataclass(obj, {"c": {}})

        assert obj == Foo1(a=1, b="b", c=Foo2(x=2, y="foo3"), d=Foo2(x=3, y="foo3"))

        assert unknown_fields == []

    def test_call_raises_error_when_there_are_invalid_overrides(self) -> None:
        obj = Foo1(a=1, b="b", c=Foo2(x=2, y=3), d=Foo2(x=3, y="foo3"))  # type: ignore[arg-type]

        overrides = {"c": 4}  # type: ignore[dict-item]

        with pytest.raises(
            FieldError,
            match=rf"^The field 'c' is expected to be of type `{Foo2}`, but is of type `{int}` instead\.$",
        ):
            update_dataclass(obj, overrides)

    def test_call_works_when_there_are_unknown_overrides(self) -> None:
        obj = Foo1(a=1, b="b", c=Foo2(x=2, y="foo3"), d=Foo2(x=3, y="foo3"))

        overrides = {"b": "a", "c": {"y": "foo4", "z": 2}, "e": 4}

        unknown_fields = update_dataclass(obj, overrides)

        assert obj == Foo1(a=1, b="a", c=Foo2(x=2, y="foo4"), d=Foo2(x=3, y="foo3"))

        assert unknown_fields == ["c.z", "e"]
