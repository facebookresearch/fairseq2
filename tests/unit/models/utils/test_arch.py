# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.models.utils.arch import ArchitectureRegistry


class TestArchitectureRegistry:
    def test_register_architecture_registers_architecture(self) -> None:
        registry = ArchitectureRegistry[str]("model")

        registry.register_architecture("arch1", lambda: "config1")
        registry.register_architecture("arch2", lambda: "config2")

        config1 = registry.get_config("arch1")
        config2 = registry.get_config("arch2")

        assert config1 == "config1"
        assert config2 == "config2"

    def test_names_returns_all_registered_names(self) -> None:
        registry = ArchitectureRegistry[str]("model")

        arch_names = {"arch1", "arch2", "arch3"}

        for arch_name in arch_names:
            registry.register_architecture(arch_name, lambda: "config")

        assert registry.names() == arch_names

    def test_register_architecture_raises_error_if_already_registered(self) -> None:
        registry = ArchitectureRegistry[str]("model")

        registry.register_architecture("arch", lambda: "config")

        with pytest.raises(
            ValueError,
            match=r"^The architecture name 'arch' is already registered for 'model'\.$",
        ):
            registry.register_architecture("arch", lambda: "config")

    def test_get_config_raises_error_if_architecture_not_registered(self) -> None:
        registry = ArchitectureRegistry[str]("model")

        with pytest.raises(
            ValueError,
            match=r"^The registry of 'model' does not contain an architecture named 'foo'\.$",
        ):
            registry.get_config("foo")
