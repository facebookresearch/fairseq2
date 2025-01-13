# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass

from fairseq2.context import Provider, get_runtime_context
from fairseq2.utils.structured import StructureError


class ConfigProcessor:
    _section_handlers: Provider[ConfigSectionHandler]

    def __init__(self, section_handlers: Provider[ConfigSectionHandler]) -> None:
        self._section_handlers = section_handlers

    def process(self, config: object) -> None:
        config_kls = type(config)

        if not is_dataclass(config_kls):
            return

        try:
            section_handler = self._section_handlers.get(config_kls)
        except LookupError:
            pass
        else:
            section_handler.process(config)

        for field in fields(config_kls):
            field_value = getattr(config, field.name)

            try:
                self.process(field_value)
            except StructureError as ex:
                raise StructureError(
                    f"`{field.name}` cannot be structured. See the nested exception for details."
                ) from ex


class ConfigSectionHandler(ABC):
    @abstractmethod
    def process(self, section: object) -> None:
        ...


def process_config(config: object) -> None:
    context = get_runtime_context()

    config_section_handlers = context.get_registry(ConfigSectionHandler)

    config_processor = ConfigProcessor(config_section_handlers)

    config_processor.process(config)
