# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pytest
import torch

from fairseq2.typing import DataType
from fairseq2.utils.dataclass import EMPTY
from fairseq2.utils.structured import (
    StructuredError,
    ValueConverter,
    is_unstructured,
    merge_unstructured,
)

# mypy: disable-error-code="arg-type"


class FooEnum(Enum):
    VALUE1 = 1
    VALUE2 = 2


@dataclass
class Foo1:
    f0: str = "foo"
    f1: int = 0
    f2: dict[str, Path] = field(default_factory=dict)
    f3: list[int] = field(default_factory=list)
    f4: Foo2 | Foo3 = field(default_factory=lambda: Foo2())
    f5: tuple[float, int] = (1.0, 2)
    f6: set[int] = field(default_factory=set)
    f7: FooEnum | None = None
    f8: DataType = torch.float32
    f9: Foo3 | None = None


@dataclass
class Foo2:
    f2_1: int = 1


@dataclass
class Foo3:
    f3_1: int = 2
    f3_2: int = 3


class TestValueConverter:
    def test_structure_works(self) -> None:
        data = {
            "f0": "abc",
            "f1": "2",
            "f2": {"a": "path1", "b": Path("path2")},
            "f3": [0, "1", 2, "3"],
            "f4": {"f3_1": "4"},
            "f5": ["3.0", "4"],
            "f6": ["1", "2", "3"],
            "f7": "VALUE2",
            "f9": {"f3_2": "4"},
        }

        value_converter = ValueConverter()

        foo = value_converter.structure(data, Foo1)

        expected_foo = Foo1(
            f0="abc",
            f1=2,
            f2={"a": Path("path1"), "b": Path("path2")},
            f3=[0, 1, 2, 3],
            f4=Foo3(f3_1=4, f3_2=3),
            f5=(3.0, 4),
            f6={1, 2, 3},
            f7=FooEnum.VALUE2,
            f8=torch.float32,
            f9=Foo3(f3_1=2, f3_2=4),
        )

        assert foo == expected_foo

    def test_structure_works_when_set_empty_is_true(self) -> None:
        data = {
            "f0": "abc",
            "f1": "2",
            "f2": {"a": "path1", "b": Path("path2")},
            "f3": [0, "1", 2, "3"],
            "f4": {"f3_1": "4"},
            "f5": ["3.0", "4"],
            "f6": ["1", "2", "3"],
            "f7": "VALUE2",
            "f9": {"f3_2": "4"},
        }

        value_converter = ValueConverter()

        foo = value_converter.structure(data, Foo1, set_empty=True)

        expected_foo = Foo1(
            f0="abc",
            f1=2,
            f2={"a": Path("path1"), "b": Path("path2")},
            f3=[0, 1, 2, 3],
            f4=Foo3(f3_1=4, f3_2=EMPTY),
            f5=(3.0, 4),
            f6={1, 2, 3},
            f7=FooEnum.VALUE2,
            f8=EMPTY,
            f9=Foo3(f3_1=EMPTY, f3_2=4),
        )

        assert foo == expected_foo

    @pytest.mark.parametrize(
        "data,kls",
        [
            ("a", int),
            ({"a": 1}, dict),
            ("a", list),
            ("a", FooEnum),
            ({"f1_1": 2, "f1_2": 3}, Foo2),
        ],
    )
    def test_structure_raises_error_when_conversion_fails(
        self, data: Any, kls: type
    ) -> None:
        value_converter = ValueConverter()

        with pytest.raises(
            StructuredError, match=rf"^`obj` cannot be structured to `{kls}`\. See nested exception for details\.$"  # fmt: skip
        ):
            value_converter.structure(data, kls)

    def test_unstructure_works(self) -> None:
        foo = Foo1(
            f0="abc",
            f1=2,
            f2={"a": Path("path1"), "b": Path("path2")},
            f3=[0, 1, 2, 3],
            f4=Foo3(f3_1=4),
            f5=(3.0, 4),
            f6={1, 2, 3},
            f7=FooEnum.VALUE2,
            f8=torch.float16,
            f9=Foo3(f3_1=1),
        )

        value_converter = ValueConverter()

        data = value_converter.unstructure(foo)

        expected_data = {
            "f0": "abc",
            "f1": 2,
            "f2": {"a": "path1", "b": "path2"},
            "f3": [0, 1, 2, 3],
            "f4": {"f3_1": 4, "f3_2": 3},
            "f5": [3.0, 4],
            "f6": [1, 2, 3],
            "f7": "VALUE2",
            "f8": "float16",
            "f9": {"f3_1": 1, "f3_2": 3},
        }

        assert data == expected_data


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
        "_del_": ["foo3"],
        "_add_": {
            "foo4": {
                "foo4_foo3": 2,
            },
            "foo7": 2.0,
            "foo8": 3,
        },
        "_set_": {
            "foo5": 2.0,
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
        "foo5": 2.0,
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
