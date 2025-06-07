# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.generation import Seq2SeqGenerator, SequenceGenerator
from fairseq2.recipe.component import ComponentManager, UnknownComponentError
from fairseq2.recipe.config import (
    Seq2SeqGeneratorSection,
    SequenceGeneratorSection,
    get_config_section,
)
from fairseq2.recipe.error import UnknownSequenceGeneratorError
from fairseq2.runtime.dependency import DependencyResolver


def _create_seq_generator(resolver: DependencyResolver) -> SequenceGenerator:
    section = get_config_section(resolver, "seq_generator", SequenceGeneratorSection)

    component_manager = resolver.resolve(ComponentManager)

    try:
        return component_manager.create_component(
            SequenceGenerator, section.name, section.config
        )
    except UnknownComponentError:
        raise UnknownSequenceGeneratorError(section.name) from None


def _create_seq2seq_generator(resolver: DependencyResolver) -> Seq2SeqGenerator:
    section = get_config_section(resolver, "seq2seq_generator", Seq2SeqGeneratorSection)

    component_manager = resolver.resolve(ComponentManager)

    try:
        return component_manager.create_component(
            Seq2SeqGenerator, section.name, section.config
        )
    except UnknownComponentError:
        raise UnknownSequenceGeneratorError(section.name) from None
