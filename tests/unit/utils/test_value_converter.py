# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import pytest
import torch

from fairseq2.typing import DataType
from fairseq2.utils.dataclass import EMPTY
from fairseq2.utils.value_converter import ValueConverter

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
    f4: Union[Foo2, Foo3] = field(default_factory=lambda: Foo2())
    f5: tuple[float, int] = (1.0, 2)
    f6: set[int] = field(default_factory=set)
    f7: Optional[FooEnum] = None
    f8: DataType = torch.float32
    f9: Optional[Foo3] = None


@dataclass
class Foo2:
    f2_1: int = 1


@dataclass
class Foo3:
    f3_1: int = 2
    f3_2: int = 3


@dataclass
class Foo4(Foo3):
    f4_1: int = 4


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
            "f8": "float16",
            "f9": {
                "_type_": "tests.unit.utils.test_value_converter.Foo4",
                "f4_1": "3",
            },
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
            f8=torch.float16,
            f9=Foo4(f4_1=3),
        )

        assert foo == expected_foo

    def test_structure_works_when_set_empty_is_true(self) -> None:
        value_converter = ValueConverter()

        foo = value_converter.structure(
            {"f1": 2, "f4": {"f3_1": 5}}, type_hint=Foo1, set_empty=True
        )

        assert foo == Foo1(
            f0=EMPTY,
            f1=2,
            f2=EMPTY,
            f3=EMPTY,
            f4=Foo3(f3_1=5, f3_2=EMPTY),
            f5=EMPTY,
            f6=EMPTY,
            f7=EMPTY,
            f8=EMPTY,
            f9=EMPTY,
        )

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
            TypeError,
            match=rf"^`obj` cannot be structured to `{kls}`\. See nested exception for details\.$",
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
            f9=Foo4(f3_1=1, f4_1=0),
        )

        value_converter = ValueConverter()

        data = value_converter.unstructure(foo, type_hint=Foo1)

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
            "f9": {
                "_type_": "tests.unit.utils.test_value_converter.Foo4",
                "f3_1": 1,
                "f3_2": 3,
                "f4_1": 0,
            },
        }

        assert data == expected_data
