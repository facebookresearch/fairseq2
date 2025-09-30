# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from fairseq2.generation import Seq2SeqGenerator, SequenceGenerator
from fairseq2.recipe.component import ComponentManager, ComponentNotKnownError
from fairseq2.recipe.config import Seq2SeqGeneratorSection, SequenceGeneratorSection
from fairseq2.recipe.error import SequenceGeneratorNotKnownError


@final
class _RecipeSequenceGeneratorFactory:
    def __init__(
        self, section: SequenceGeneratorSection, component_manager: ComponentManager
    ) -> None:
        self._section = section
        self._component_manager = component_manager

    def create(self) -> SequenceGenerator:
        section = self._section

        try:
            return self._component_manager.create_component(
                SequenceGenerator, section.name, section.config
            )
        except ComponentNotKnownError:
            raise SequenceGeneratorNotKnownError(section.name) from None


@final
class _RecipeSeq2SeqGeneratorFactory:
    def __init__(
        self, section: Seq2SeqGeneratorSection, component_manager: ComponentManager
    ) -> None:
        self._section = section
        self._component_manager = component_manager

    def create(self) -> Seq2SeqGenerator:
        section = self._section

        try:
            return self._component_manager.create_component(
                Seq2SeqGenerator, section.name, section.config
            )
        except ComponentNotKnownError:
            raise SequenceGeneratorNotKnownError(section.name) from None
