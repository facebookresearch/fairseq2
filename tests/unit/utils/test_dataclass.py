# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from fairseq2.utils.dataclass import EMPTY, merge_dataclass


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


def test_merge_dataclass() -> None:
    target = Foo1(a=3, b=Foo3(y=5), c="foo")
    source = Foo1(a=2, b=Foo3(y=EMPTY, z=3), c=EMPTY)  # type: ignore[arg-type]

    target = merge_dataclass(target, source)

    assert target == Foo1(a=2, b=Foo3(y=5, z=3), c="foo")

    target = Foo1(a=3, b=Foo3(y=1), c="foo")
    source = Foo1(a=EMPTY, b=Foo3(y=2, z=EMPTY), c="foo")  # type: ignore[arg-type]

    target = merge_dataclass(target, source)

    assert target == Foo1(a=3, b=Foo3(y=2, z=2), c="foo")

    target = Foo1(a=3, b=Foo2(x=1), c="foo")
    source = Foo1(a=2, b=EMPTY, c="foo")  # type: ignore[arg-type]

    target = merge_dataclass(target, source)

    assert target == Foo1(a=2, b=Foo2(x=1), c="foo")
