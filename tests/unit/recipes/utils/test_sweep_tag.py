# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Hashable

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

        tagger = self._create_tagger()

        tag = tagger.generate("foo", config)

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

        tagger = self._create_tagger()

        tag = tagger.generate("foo", config)

        assert tag == "ps_foo.ws_2.a618ea54"

    def test_call_works_when_keys_are_disallowed(self) -> None:
        config = {
            "foo1": "a",
            "foo2": {"foo1": 0.2},
            "foo3": True,
            "foo4": 1,
            "foo5": 2.0,
            "foo6": [1, 2, 3, {"foo7": "a"}],
            "foo8": "b",  # should be ignored.
            "foo9": "c",  # should be ignored.
        }

        tagger = self._create_tagger()

        tag = tagger.generate("foo", config)

        assert tag == "ps_foo.ws_2.a618ea54"

    def test_call_works_when_sweep_format_is_specified(self) -> None:
        fmt = "ps_{preset}.{{foo9}}.foo5_{{{foo5}}}.foo21_{foo2.foo1}.foo61_{foo6[1]}.{hash}"

        config = {
            "foo1": "a",
            "foo5": 2.0,
            "foo2": {"foo1": 0.2},
            "foo4": 1,
            "foo3": True,
            "foo6": [1, 2, 3],
        }

        tagger = self._create_tagger(fmt)

        tag = tagger.generate("foo", config)

        assert tag == "ps_foo.{foo9}.foo5_{2.0}.foo21_0.2.foo61_2.a618ea54"

    def test_call_raises_error_when_sweep_format_is_invalid(self) -> None:
        fmt = "foo_{foo1"

        with pytest.raises(
            ValueError, match=r"^`fmt` must have matching opening and closing braces.$"  # fmt: skip
        ):
            self._create_tagger(fmt)

    def test_call_raises_error_when_sweep_format_has_unknown_key(self) -> None:
        fmt = "foo_{foo2}.foo_{foo3}.foo_{foo2}"

        config = {"foo1": "a"}

        tagger = self._create_tagger(fmt)

        with pytest.raises(
            SweepFormatPlaceholderError, match=r"^The sweep format string must contain only placeholders that correspond to the configuration keys, but contains the following unexpected placeholder\(s\): foo2, foo3$"  # fmt: skip
        ):
            tagger.generate("foo", config)

    @staticmethod
    def _create_tagger(fmt: str | None = None) -> SweepTagGenerator:
        allowed_keys: set[Hashable] = {f"foo{i}" for i in range(7)}

        return SweepTagGenerator(2, allowed_keys, fmt)
