# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping

import pytest

from fairseq2.utils.config import ReplaceEnvDirective, StandardConfigMerger
from tests.unit.helper import FooEnvironment


def test_config_merge_works() -> None:
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
        "foo9": {
            "foo10": {"foo11": 3},
        },
    }

    source = {
        "_del_": ["foo3"],
        "_set_": {
            "foo5": 2.0,
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
                "foo4_foo3": "y",
            },
            "foo4_foo4": "z",
        },
        "foo6": 1.0,
        "foo7": {
            "foo8": {"foo9": 1},
        },
        "foo9": {
            "_set_": {"foo10": 2},
        },
    }

    merger = StandardConfigMerger()

    output = merger.merge(target, source)

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
            "foo4_foo3": "y",
            "foo4_foo4": "z",
        },
        "foo5": 2.0,
        "foo6": 1.0,
        "foo7": {
            "foo8": {"foo9": 1},
        },
        "foo9": {"foo10": 2},
    }

    assert output == expected_output


def test_config_merge_raises_error_when_type_is_invalid() -> None:
    target: object
    source: object

    target = {}
    source = {"_del_": "foo"}

    merger = StandardConfigMerger()

    with pytest.raises(
        TypeError, match=rf"^_del_ at `overrides` must be of type `{list}`, but is of type `{str}` instead\.$"  # fmt: skip
    ):
        merger.merge(target, source)

    target = {"foo1": {"foo2": {}}}
    source = {"foo1": {"foo2": {"_del_": "foo"}}}

    with pytest.raises(
        TypeError, match=rf"^foo1\.foo2\._del_ at `overrides` must be of type `{list}`, but is of type `{str}` instead\.$"  # fmt: skip
    ):
        merger.merge(target, source)

    target = {"foo1": {"foo2": {}}}
    source = {"foo1": {"foo2": {"_del_": [0]}}}

    with pytest.raises(
        TypeError, match=rf"^Each element under foo1\.foo2\._del_ at `overrides` must be of type `{str}`, but the element at index 0 is of type `{int}` instead\.$"  # fmt: skip
    ):
        merger.merge(target, source)

    target = {"foo1": {"foo2": {}}}
    source = {"foo1": {"foo2": {"_set_": "foo"}}}

    with pytest.raises(
        TypeError, match=rf"^foo1\.foo2\._set_ at `overrides` must be of type `{Mapping}`, but is of type `{str}` instead\.$"  # fmt: skip
    ):
        merger.merge(target, source)

    target = {"foo1": {"foo2": {}}}
    source = {"foo1": {"foo2": {"_set_": {0: "foo"}}}}

    with pytest.raises(
        TypeError, match=rf"^Each key under foo1\.foo2\._set_ at `overrides` must be of type `{str}`, but the key at index 0 is of type `{int}` instead\.$"  # fmt: skip
    ):
        merger.merge(target, source)


def test_replace_env_directive_works() -> None:
    env = FooEnvironment({"FOO1": "f001", "FOO2": "f002"})

    directive = ReplaceEnvDirective(env)

    output = directive.execute(
        "abc ${env:FOO1} xyz ${env:FOO2} ijk ${env:FOO3:f003} ${env:FOO4}", None
    )

    assert output == "abc f001 xyz f002 ijk f003 "
