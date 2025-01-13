# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from fairseq2.config_registry import ConfigRegistry
from fairseq2.error import AlreadyExistsError


@dataclass
class Foo:
    x: str


class TestConfigRegistry:
    def test_register_works(self) -> None:
        registry = ConfigRegistry(Foo)

        registry.register("name1", lambda: Foo("config1"))
        registry.register("name2", lambda: Foo("config2"))

        config1 = registry.get("name1")
        config2 = registry.get("name2")

        assert config1.x == "config1"
        assert config2.x == "config2"

    def test_names_works(self) -> None:
        registry = ConfigRegistry(Foo)

        names = {"name1", "name2", "name3"}

        for name in names:
            registry.register(name, lambda: Foo("config"))

        assert registry.names() == names

    def test_register_raises_error_when_name_is_already_registered(
        self,
    ) -> None:
        registry = ConfigRegistry(Foo)

        registry.register("name", lambda: Foo("config"))

        with pytest.raises(
            AlreadyExistsError, match=r"^The registry has already a configuration named 'name'\.$",  # fmt: skip
        ):
            registry.register("name", lambda: Foo("config"))

    def test_get_raises_error_when_name_is_not_registered(self) -> None:
        registry = ConfigRegistry(Foo)

        with pytest.raises(
            LookupError, match=r"^'foo' is not a registered configuration name\.$",  # fmt: skip
        ):
            registry.get("foo")
