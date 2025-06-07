# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.dependency import DependencyResolver
from fairseq2.generation import Seq2SeqGenerator, SequenceGenerator
from fairseq2.recipe.component import resolve_component
from fairseq2.recipe.config import ComponentSection, get_recipe_config_section


def create_seq_generator(resolver: DependencyResolver) -> SequenceGenerator:
    section = get_recipe_config_section(resolver, "seq_generator", ComponentSection)

    return resolve_component(resolver, SequenceGenerator, section.name, section.config)


def create_seq2seq_generator(resolver: DependencyResolver) -> Seq2SeqGenerator:
    section = get_recipe_config_section(resolver, "seq_generator", ComponentSection)

    return resolve_component(resolver, Seq2SeqGenerator, section.name, section.config)
