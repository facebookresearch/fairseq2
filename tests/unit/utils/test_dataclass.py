# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from fairseq2.utils.dataclass import EMPTY, fill_empty_fields


@dataclass
class Foo1:
    a: int
    b: Foo2 | Foo3
    c: str


@dataclass
class Foo2:
    x: int


@dataclass
class Foo3:
    y: int = 1
    z: int = 2


def test_fill_empty_fields_works() -> None:
    target = Foo1(a=2, b=Foo3(y=EMPTY, z=3), c=EMPTY)  # type: ignore[arg-type]
    source = Foo1(a=3, b=Foo3(y=5), c="foo")

    fill_empty_fields(target, source)

    assert target == Foo1(a=2, b=Foo3(y=5, z=3), c="foo")

    target = Foo1(a=EMPTY, b=Foo3(y=2, z=EMPTY), c="foo")  # type: ignore[arg-type]
    source = Foo1(a=3, b=Foo3(y=1), c="foo")

    fill_empty_fields(target, source)

    assert target == Foo1(a=3, b=Foo3(y=2, z=2), c="foo")

    target = Foo1(a=2, b=EMPTY, c="foo")  # type: ignore[arg-type]
    source = Foo1(a=3, b=Foo2(x=1), c="foo")

    fill_empty_fields(target, source)

    assert target == Foo1(a=2, b=Foo2(x=1), c="foo")


def test_fill_empty_fields_raises_error_when_types_mismatch() -> None:
    target = Foo1(a=3, b=Foo2(x=1), c="foo")
    source = Foo3()

    with pytest.raises(
        TypeError, match=rf"^`target` and `source` must be of the same type, but they are of types `{Foo1}` and `{Foo3}` instead\.$",  # fmt: skip
    ):
        fill_empty_fields(target, source)


def test_fill_empty_fields_raises_error_when_empty_field_remains() -> None:
    target = Foo1(a=1, b=Foo2(x=EMPTY), c="foo")  # type: ignore[arg-type]
    source = Foo1(a=2, b=Foo3(), c="foo")

    with pytest.raises(
        ValueError, match=r"^`target` must have no empty field after `fill_empty_fields\(\)`, but one or more fields remained empty\.$",  # fmt: skip
    ):
        fill_empty_fields(target, source)
