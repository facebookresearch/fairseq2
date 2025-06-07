# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.dependency import DependencyResolver
from fairseq2.generation import Seq2SeqGenerator, SequenceGenerator
from fairseq2.recipe.component import resolve_component
from fairseq2.recipe.config import (
    Seq2SeqGeneratorSection,
    SequenceGeneratorSection,
    get_config_section,
)
from fairseq2.utils.structured import StructureError


def create_seq_generator(resolver: DependencyResolver) -> SequenceGenerator:
    section = get_config_section(resolver, "seq_generator", SequenceGeneratorSection)

    try:
        return resolve_component(
            resolver, SequenceGenerator, section.name, section.config
        )
    except StructureError as ex:
        raise StructureError(
            f"The '{section.name}' sequence generator configuration cannot be parsed. See the nested exception for details."
        ) from ex


def create_seq2seq_generator(resolver: DependencyResolver) -> Seq2SeqGenerator:
    section = get_config_section(resolver, "seq2seq_generator", Seq2SeqGeneratorSection)

    try:
        return resolve_component(
            resolver, Seq2SeqGenerator, section.name, section.config
        )
    except StructureError as ex:
        raise StructureError(
            f"The '{section.name}' sequence generator configuration cannot be parsed. See the nested exception for details."
        ) from ex
