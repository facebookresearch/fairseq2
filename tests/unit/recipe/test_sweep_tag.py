# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from fairseq2.recipe.config import CommonSection
from fairseq2.recipe.internal.config import _RecipeConfigHolder
from fairseq2.recipe.internal.sweep_tag import _StandardSweepTagGenerator
from fairseq2.utils.structured import StandardValueConverter
from fairseq2.utils.validation import ValidationError
from fairseq2.world_info import WorldInfo


class TestSweepTagGenerator:
    def test_call_works(self) -> None:
        section = CommonSection()

        world_info = WorldInfo(0, 2, 0, 1)

        config = {
            "foo1": "a",
            "foo2": {"foo1": 0.2},
            "foo3": True,
            "foo4": 1,
            "foo5": 2.0,
            "foo6": [1, 2, 3],
        }

        value_converter = StandardValueConverter()

        generator = _StandardSweepTagGenerator(
            section, world_info, _RecipeConfigHolder(config), value_converter
        )

        tag = generator.maybe_generate()

        assert tag == "ws_2.0dde6d9f"

    def test_call_works_when_key_order_is_different(self) -> None:
        section = CommonSection()

        world_info = WorldInfo(0, 2, 0, 1)

        config = {
            "foo1": "a",
            "foo5": 2.0,
            "foo2": {"foo1": 0.2},
            "foo4": 1,
            "foo3": True,
            "foo6": [1, 2, 3],
        }

        value_converter = StandardValueConverter()

        generator = _StandardSweepTagGenerator(
            section, world_info, _RecipeConfigHolder(config), value_converter
        )

        tag = generator.maybe_generate()

        assert tag == "ws_2.0dde6d9f"

    def test_call_works_when_sweep_format_is_specified(self) -> None:
        section = CommonSection(
            sweep_format="{{foo9}}.foo5_{{{foo5}}}.foo21_{foo2.foo1}.foo61_{foo6[1]}.{hash}"
        )

        world_info = WorldInfo(0, 2, 0, 1)

        config = {
            "foo1": "a",
            "foo5": 2.0,
            "foo2": {"foo1": 0.2},
            "foo4": 1,
            "foo3": True,
            "foo6": [1, 2, 3],
        }

        value_converter = StandardValueConverter()

        generator = _StandardSweepTagGenerator(
            section, world_info, _RecipeConfigHolder(config), value_converter
        )

        tag = generator.maybe_generate()

        assert tag == "{foo9}.foo5_{2.0}.foo21_0.2.foo61_2.0dde6d9f"

    def test_call_raises_error_when_sweep_format_is_invalid(self) -> None:
        section = CommonSection(sweep_format="foo_{foo1")

        world_info = WorldInfo(0, 2, 0, 1)

        config: dict[str, object] = {}

        value_converter = StandardValueConverter()

        generator = _StandardSweepTagGenerator(
            section, world_info, _RecipeConfigHolder(config), value_converter
        )

        with pytest.raises(
            ValidationError, match=r"^`common` is not valid: `sweep_format` must have matching opening and closing braces.$"  # fmt: skip
        ):
            generator.maybe_generate()

    def test_call_raises_error_when_sweep_format_has_unknown_key(self) -> None:
        section = CommonSection(sweep_format="foo_{foo2}.foo_{foo3}.foo_{foo2}")

        world_info = WorldInfo(0, 2, 0, 1)

        config = {"foo1": "a"}

        value_converter = StandardValueConverter()

        generator = _StandardSweepTagGenerator(
            section, world_info, _RecipeConfigHolder(config), value_converter
        )

        with pytest.raises(
            ValidationError, match=r"^`common` is not valid: `sweep_format` must contain only placeholders that correspond to the configuration keys, but contains unexpected placeholder\(s\) foo2, foo3\.$"  # fmt: skip
        ):
            generator.maybe_generate()
