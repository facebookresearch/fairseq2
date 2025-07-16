# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pytest

from fairseq2.typing import EMPTY
from fairseq2.utils.merge import MergeError, merge_dataclass, merge_map


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


def test_merge_object_works() -> None:
    target = {
        "foo1": "abc",
        "foo2": {
            "foo2_foo1": 4,
            "foo2_foo2": {
                "foo2_foo2_foo1": "x",
            },
            "foo2_foo3": 4,
        },
        "foo3": True,
        "foo4": {
            "foo4_foo1": "y",
            "foo4_foo2": "z",
        },
        "foo5": 1.0,
    }

    source = {
        "_del_": ["foo3"],
        "_set_": {
            "foo5": 2.0,
            "foo6": 1.0,
        },
        "foo2": {
            "_del_": ["foo2_foo1"],
            "_set_": {
                "foo2_foo4": "a",
            },
        },
        "foo4": {
            "_set_": {
                "foo4_foo1": "x",
            }
        },
    }

    output = merge_map(target, source)

    expected_output = {
        "foo1": "abc",
        "foo2": {
            "foo2_foo2": {
                "foo2_foo2_foo1": "x",
            },
            "foo2_foo3": 4,
            "foo2_foo4": "a",
        },
        "foo4": {
            "foo4_foo1": "x",
            "foo4_foo2": "z",
        },
        "foo5": 2.0,
        "foo6": 1.0,
    }

    assert output == expected_output


def test_merge_map_raises_error_when_type_is_invalid() -> None:
    target: object
    source: object

    target = {"foo1": 0}
    source = {"foo1": 1}

    with pytest.raises(
        MergeError, match=rf"^The 'foo1' path at `source` must be of type `{Mapping}`, but is of type `{int}` instead\."  # fmt: skip
    ):
        merge_map(target, source)

    target = {"foo1": 1}
    source = {"foo1": {"foo2": 1}}

    with pytest.raises(
        MergeError, match=rf"^The 'foo1' path at `target` must be of type `{Mapping}`, but is of type `{int}` instead\."  # fmt: skip
    ):
        merge_map(target, source)

    target = {}
    source = {"_del_": "foo"}

    with pytest.raises(
        MergeError, match=rf"^'_del_' at `source` must be of type `{list}`, but is of type `{str}` instead\.$"  # fmt: skip
    ):
        merge_map(target, source)

    target = {"foo1": {"foo2": {}}}
    source = {"foo1": {"foo2": {"_del_": "foo"}}}

    with pytest.raises(
        MergeError, match=rf"^'foo1\.foo2\._del_' at `source` must be of type `{list}`, but is of type `{str}` instead\.$"  # fmt: skip
    ):
        merge_map(target, source)

    target = {"foo1": {"foo2": {}}}
    source = {"foo1": {"foo2": {"_del_": [0]}}}

    with pytest.raises(
        MergeError, match=rf"^Each element under 'foo1\.foo2\._del_' at `source` must be of type `str`, but the element at index 0 is of type `{int}` instead\.$"  # fmt: skip
    ):
        merge_map(target, source)

    target = {"foo1": {"foo2": {}}}
    source = {"foo1": {"foo2": {"_set_": "foo"}}}

    with pytest.raises(
        MergeError, match=rf"^'foo1\.foo2\._set_' at `source` must be of type `{Mapping}`, but is of type `{str}` instead\.$"  # fmt: skip
    ):
        merge_map(target, source)

    target = {"foo1": {"foo2": {}}}
    source = {"foo1": {"foo2": {"_set_": {0: "foo"}}}}

    with pytest.raises(
        MergeError, match=rf"^Each key under 'foo1\.foo2\._set_' at `source` must be of type `str`, but the key at index 0 is of type `{int}` instead\.$"  # fmt: skip
    ):
        merge_map(target, source)


def test_merge_map_raises_error_when_path_does_not_exist() -> None:
    target: object
    source: object

    target = {"foo1": 0}
    source = {"foo2": 1}

    with pytest.raises(
        MergeError, match=r"^`target` must have an item at path 'foo2'\."  # fmt: skip
    ):
        merge_map(target, source)
