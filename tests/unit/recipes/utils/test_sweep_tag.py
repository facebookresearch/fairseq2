# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from fairseq2.recipes.utils.sweep_tag import (
    SweepFormatPlaceholderError,
    SweepTagGenerator,
)


class TestSweepTagGenerator:
    def test_call_works(self) -> None:
        config = {
            "foo1": "a",
            "foo2": {"foo1": 0.2},
            "foo3": True,
            "foo4": 1,
            "foo5": 2.0,
            "foo6": [1, 2, 3],
        }

        tag_generator = SweepTagGenerator(world_size=2)

        tag = tag_generator.generate("foo", config)

        assert tag == "ps_foo.ws_2.a618ea54"

    def test_call_works_when_key_order_is_different(self) -> None:
        config = {
            "foo1": "a",
            "foo5": 2.0,
            "foo2": {"foo1": 0.2},
            "foo4": 1,
            "foo3": True,
            "foo6": [1, 2, 3],
        }

        tag_generator = SweepTagGenerator(world_size=2)

        tag = tag_generator.generate("foo", config)

        assert tag == "ps_foo.ws_2.a618ea54"

    def test_call_works_when_sweep_format_is_specified(self) -> None:
        world_size = 2

        sweep_format = "ps_{preset}.{{foo9}}.foo5_{{{foo5}}}.foo21_{foo2.foo1}.foo61_{foo6[1]}.{hash}"

        config = {
            "foo1": "a",
            "foo5": 2.0,
            "foo2": {"foo1": 0.2},
            "foo4": 1,
            "foo3": True,
            "foo6": [1, 2, 3],
        }

        tag_generator = SweepTagGenerator(world_size, sweep_format)

        tag = tag_generator.generate("foo", config)

        assert tag == "ps_foo.{foo9}.foo5_{2.0}.foo21_0.2.foo61_2.a618ea54"

    def test_call_raises_error_when_sweep_format_is_invalid(self) -> None:
        world_size = 2

        sweep_format = "foo_{foo1"

        with pytest.raises(
            ValueError, match=r"^`fmt` must have matching opening and closing braces.$"  # fmt: skip
        ):
            SweepTagGenerator(world_size, sweep_format)

    def test_call_raises_error_when_sweep_format_has_unknown_key(self) -> None:
        world_size = 2

        sweep_format = "foo_{foo2}.foo_{foo3}.foo_{foo2}"

        config = {"foo1": "a"}

        tag_generator = SweepTagGenerator(world_size, sweep_format)

        with pytest.raises(
            SweepFormatPlaceholderError, match=r"^The sweep format string must contain only placeholders that correspond to the configuration keys, but contains the following unexpected placeholder\(s\): foo2, foo3$"  # fmt: skip
        ):
            tag_generator.generate("foo", config)
