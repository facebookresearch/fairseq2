# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, final

from typing_extensions import override

from fairseq2.dependency import DependencyContainer
from fairseq2.utils.structured import StructuredError, ValueConverter


class ConfigManager(ABC):
    @abstractmethod
    def get_config(self, path: str, type_expr: Any) -> Any:
        ...


@final
class StandardConfigManager(ConfigManager):
    _value_converter: ValueConverter
    _config_dict: dict[str, object]

    def __init__(self, value_converter: ValueConverter) -> None:
        self._value_converter = value_converter

        self._config_dict = {}

    def update_config_dict(self, config_dict: dict[str, object]) -> None:
        self._config_dict.update(config_dict)

    @override
    def get_config(self, path: str, type_expr: Any) -> Any:
        try:
            config = self._config_dict[path]
        except KeyError:
            raise ConfigNotFoundError(
                f"The '{path}' configuration is not found."
            ) from None

        try:
            return self._value_converter.structure(config, type_expr)
        except StructuredError as ex:
            raise ConfigError(
                f"The '{path}' configuration cannot be parsed. See nested exception for details."
            ) from ex


class ConfigError(RuntimeError):
    pass


class ConfigNotFoundError(ConfigError):
    pass


def register_config_manager(container: DependencyContainer) -> None:
    container.register(StandardConfigManager)

    container.register_factory(
        ConfigManager, lambda r: r.resolve(StandardConfigManager)
    )
