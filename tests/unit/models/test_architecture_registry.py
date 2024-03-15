# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.models.architecture_registry import ModelArchitectureRegistry


class TestModelArchitectureRegistry:
    def test_register_works(self) -> None:
        registry = ModelArchitectureRegistry[str]()

        registry.register("arch1", lambda: "config1")
        registry.register("arch2", lambda: "config2")

        config1 = registry.get_config("arch1")
        config2 = registry.get_config("arch2")

        assert config1 == "config1"
        assert config2 == "config2"

    def test_names_works(self) -> None:
        registry = ModelArchitectureRegistry[str]()

        archs = {"arch1", "arch2", "arch3"}

        for arch in archs:
            registry.register(arch, lambda: "config")

        assert registry.names() == archs

    def test_register_raises_error_when_architecture_is_already_registered(
        self,
    ) -> None:
        registry = ModelArchitectureRegistry[str]()

        registry.register("arch", lambda: "config")

        with pytest.raises(
            ValueError,
            match=r"^`arch` must be a unique model architecture, but 'arch' is already registered\.$",
        ):
            registry.register("arch", lambda: "config")

    def test_get_config_raises_error_when_architecture_is_not_registered(self) -> None:
        registry = ModelArchitectureRegistry[str]()

        with pytest.raises(
            ValueError,
            match=r"^`arch` must be a registered model architecture, but is 'foo' instead\.$",
        ):
            registry.get_config("foo")
