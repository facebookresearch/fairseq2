# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import pytest

from fairseq2.utils.structured import (
    StructuredError,
    is_unstructured,
    merge_unstructured,
)


def test_is_unstructured_works_when_object_is_unstructed() -> None:
    obj = {
        "foo1": True,
        "foo2": 1,
        "foo3": 1.0,
        "foo4": "a",
        "foo5": {
            "foo6": "x",
        },
        "foo7": [1, False, 3.0, "a"],
        "foo8": None,
    }

    assert is_unstructured(obj)


def test_is_unstructured_works_when_object_is_structed() -> None:
    obj: Any

    obj = object()

    assert not is_unstructured(obj)

    obj = {
        "foo1": True,
        "foo2": object(),
        "foo3": "a",
    }

    assert not is_unstructured(obj)


def test_merge_unstructured_works() -> None:
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
        "foo6": [1, 2, 3],
    }

    source = {
        "_del_": ["foo3", "foo5"],
        "_add_": {
            "foo4": {
                "foo4_foo3": 2,
            },
            "foo7": 2.0,
            "foo8": 3,
        },
        "foo2": {
            "_del_": ["foo2_foo1"],
            "_add_": {
                "foo3_foo4": "a",
            },
            "foo2_foo2": {
                "foo2_foo2_foo1": "b",
            },
            "foo2_foo3": 5,
        },
        "foo6": [4, 5, 6],
    }

    output = merge_unstructured(target, source)

    expected_output = {
        "foo1": "abc",
        "foo2": {
            "foo2_foo2": {
                "foo2_foo2_foo1": "b",
            },
            "foo2_foo3": 5,
            "foo3_foo4": "a",
        },
        "foo4": {
            "foo4_foo3": 2,
        },
        "foo6": [4, 5, 6],
        "foo7": 2.0,
        "foo8": 3,
    }

    assert output == expected_output


def test_merge_unstructured_raises_error_when_type_is_invalid() -> None:
    target: Any
    source: Any

    target = object()
    source = None

    with pytest.raises(
        StructuredError, match=r"^`target` must be a composition of types `bool`, `int`, `float`, `str`, `list`, and `dict`\.$"  # fmt: skip
    ):
        merge_unstructured(target, source)

    target = None
    source = object()

    with pytest.raises(
        StructuredError, match=r"^`source` must be a composition of types `bool`, `int`, `float`, `str`, `list`, and `dict`\.$"  # fmt: skip
    ):
        merge_unstructured(target, source)

    target = {}
    source = None

    with pytest.raises(
        StructuredError, match=r"^`target` is of type `dict`, but `source` is of type `NoneType`\.$"  # fmt: skip
    ):
        merge_unstructured(target, source)

    target = None
    source = {}

    with pytest.raises(
        StructuredError, match=r"^`target` is of type `NoneType`, but `source` is of type `dict`\.$"  # fmt: skip
    ):
        merge_unstructured(target, source)

    target = {"foo1": {"foo2": {}}}
    source = {"foo1": {"foo2": 10}}

    with pytest.raises(
        StructuredError, match=r"^'foo1\.foo2' is of type `dict` in `target`, but is of type `int` in `source`\.$"  # fmt: skip
    ):
        merge_unstructured(target, source)

    target = {"foo1": {"foo2": 10}}
    source = {"foo1": {"foo2": {}}}

    with pytest.raises(
        StructuredError, match=r"^'foo1\.foo2' is of type `int` in `target`, but is of type `dict` in `source`\.$"  # fmt: skip
    ):
        merge_unstructured(target, source)

    target = {}
    source = {"_del_": "foo"}

    with pytest.raises(
        StructuredError, match=r"^'_del_' in `source` must be of type `list`, but is of type `str` instead\.$"  # fmt: skip
    ):
        merge_unstructured(target, source)

    target = {"foo1": {"foo2": {}}}
    source = {"foo1": {"foo2": {"_del_": "foo"}}}

    with pytest.raises(
        StructuredError, match=r"^'foo1\.foo2\._del_' in `source` must be of type `list`, but is of type `str` instead\.$"  # fmt: skip
    ):
        merge_unstructured(target, source)

    target = {"foo1": {"foo2": {}}}
    source = {"foo1": {"foo2": {"_del_": [0]}}}

    with pytest.raises(
        StructuredError, match=r"^Each element under 'foo1\.foo2\._del_' in `source` must be of type `str`, but the element at index 0 is of type `int` instead\.$"  # fmt: skip
    ):
        merge_unstructured(target, source)

    target = {"foo1": {"foo2": {}}}
    source = {"foo1": {"foo2": {"_add_": "foo"}}}

    with pytest.raises(
        StructuredError, match=r"^'foo1\.foo2\._add_' in `source` must be of type `dict`, but is of type `str` instead\.$"  # fmt: skip
    ):
        merge_unstructured(target, source)

    target = {"foo1": {"foo2": {}}}
    source = {"foo1": {"foo2": {"_add_": {0: "foo"}}}}

    with pytest.raises(
        StructuredError, match=r"^Each key under 'foo1\.foo2\._add_' in `source` must be of type `str`, but the key at index 0 is of type `int` instead\.$"  # fmt: skip
    ):
        merge_unstructured(target, source)


def test_merge_unstructured_raises_error_when_path_does_not_exist() -> None:
    target: Any
    source: Any

    target = {"foo1": 0}
    source = {"foo2": 1}

    with pytest.raises(
        ValueError, match=r"^`target` must contain a 'foo2' key since it exists in `source`\."  # fmt: skip
    ):
        merge_unstructured(target, source)

    target = {"foo1": {"foo2": 0}}
    source = {"foo1": {"foo3": 1}}

    with pytest.raises(
        ValueError, match=r"^`target` must contain a 'foo1\.foo3' key since it exists in `source`\."  # fmt: skip
    ):
        merge_unstructured(target, source)
