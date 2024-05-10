# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import pytest

from fairseq2.config_registry import ConfigRegistry


@dataclass
class Foo:
    x: str


class TestConfigRegistry:
    def test_register_works(self) -> None:
        registry = ConfigRegistry[Foo]()

        registry.register("name1", lambda: Foo("config1"))
        registry.register("name2", lambda: Foo("config2"))

        config1 = registry.get("name1")
        config2 = registry.get("name2")

        assert config1.x == "config1"
        assert config2.x == "config2"

    def test_names_works(self) -> None:
        registry = ConfigRegistry[Foo]()

        names = {"name1", "name2", "name3"}

        for name in names:
            registry.register(name, lambda: Foo("config"))

        assert registry.names() == names

    def test_register_raises_error_when_name_is_already_registered(
        self,
    ) -> None:
        registry = ConfigRegistry[Foo]()

        registry.register("name", lambda: Foo("config"))

        with pytest.raises(
            ValueError,
            match=r"^`name` must be a unique configuration name, but 'name' has already a registered configuration factory\.$",
        ):
            registry.register("name", lambda: Foo("config"))

    def test_get_raises_error_when_name_is_not_registered(self) -> None:
        registry = ConfigRegistry[Foo]()

        with pytest.raises(
            ValueError,
            match=r"^`name` must be a registered configuration name, but is 'foo' instead\.$",
        ):
            registry.get("foo")
