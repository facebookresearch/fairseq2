# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from fairseq2.recipe.utils.sweep_tag import (
    SweepFormatError,
    SweepFormatPlaceholderError,
    SweepTagGenerator,
)
from fairseq2.utils.structured import StandardValueConverter


class TestSweepTagGenerator:
    def test_call_works(self) -> None:
        value_converter = StandardValueConverter()

        tag_generator = SweepTagGenerator(value_converter)

        config = {
            "foo1": "a",
            "foo2": {"foo1": 0.2},
            "foo3": True,
            "foo4": 1,
            "foo5": 2.0,
            "foo6": [1, 2, 3],
        }

        tag = tag_generator.generate(
            config, world_size=2, sweep_format="ws_{world_size}.{hash}"
        )

        assert tag == "ws_2.0dde6d9f"

    def test_call_works_when_key_order_is_different(self) -> None:
        value_converter = StandardValueConverter()

        tag_generator = SweepTagGenerator(value_converter)

        config = {
            "foo1": "a",
            "foo5": 2.0,
            "foo2": {"foo1": 0.2},
            "foo4": 1,
            "foo3": True,
            "foo6": [1, 2, 3],
        }

        tag = tag_generator.generate(
            config, world_size=2, sweep_format="ws_{world_size}.{hash}"
        )

        assert tag == "ws_2.0dde6d9f"

    def test_call_works_when_sweep_format_is_specified(self) -> None:
        value_converter = StandardValueConverter()

        tag_generator = SweepTagGenerator(value_converter)

        config = {
            "foo1": "a",
            "foo5": 2.0,
            "foo2": {"foo1": 0.2},
            "foo4": 1,
            "foo3": True,
            "foo6": [1, 2, 3],
        }

        world_size = 2

        sweep_format = (
            "{{foo9}}.foo5_{{{foo5}}}.foo21_{foo2.foo1}.foo61_{foo6[1]}.{hash}"
        )

        tag = tag_generator.generate(config, world_size, sweep_format)

        assert tag == "{foo9}.foo5_{2.0}.foo21_0.2.foo61_2.0dde6d9f"

    def test_call_raises_error_when_sweep_format_is_invalid(self) -> None:
        value_converter = StandardValueConverter()

        tag_generator = SweepTagGenerator(value_converter)

        config: dict[str, object] = {}

        world_size = 2

        sweep_format = "foo_{foo1"

        with pytest.raises(
            SweepFormatError, match=r"^`sweep_format` must have matching opening and closing braces.$"  # fmt: skip
        ):
            tag_generator.generate(config, world_size, sweep_format)

    def test_call_raises_error_when_sweep_format_has_unknown_key(self) -> None:
        value_converter = StandardValueConverter()

        tag_generator = SweepTagGenerator(value_converter)

        config = {"foo1": "a"}

        world_size = 2

        sweep_format = "foo_{foo2}.foo_{foo3}.foo_{foo2}"

        with pytest.raises(
            SweepFormatPlaceholderError, match=r"^`sweep_format` must contain only placeholders that correspond to the configuration keys, but contains the following unexpected placeholder\(s\): foo2, foo3$"  # fmt: skip
        ):
            tag_generator.generate(config, world_size, sweep_format)
