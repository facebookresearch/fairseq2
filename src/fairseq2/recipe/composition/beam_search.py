# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from types import NoneType

from fairseq2.generation import Seq2SeqGenerator, SequenceGenerator
from fairseq2.generation.beam_search import (
    BeamSearchAlgorithm,
    StandardBeamSearchAlgorithm,
)
from fairseq2.recipe.component import register_component
from fairseq2.recipe.config import (
    BEAM_SEARCH_GENERATOR,
    STANDARD_BEAM_SEARCH_ALGO,
    BeamSearchConfig,
)
from fairseq2.recipe.internal.beam_search import (
    _BeamSearchSeq2SeqGeneratorFactory,
    _BeamSearchSequenceGeneratorFactory,
)
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


def _register_beam_search(container: DependencyContainer) -> None:
    # Sequence
    def create_seq_generator(
        resolver: DependencyResolver, config: BeamSearchConfig
    ) -> SequenceGenerator:
        gen_factory = resolver.resolve(_BeamSearchSequenceGeneratorFactory)

        return gen_factory.create(config)

    register_component(
        container,
        SequenceGenerator,
        BEAM_SEARCH_GENERATOR,
        config_kls=BeamSearchConfig,
        factory=create_seq_generator,
    )

    container.register_type(_BeamSearchSequenceGeneratorFactory)

    # Seq2Seq
    def create_seq2seq_generator(
        resolver: DependencyResolver, config: BeamSearchConfig
    ) -> Seq2SeqGenerator:
        gen_factory = resolver.resolve(_BeamSearchSeq2SeqGeneratorFactory)

        return gen_factory.create(config)

    register_component(
        container,
        Seq2SeqGenerator,
        BEAM_SEARCH_GENERATOR,
        config_kls=BeamSearchConfig,
        factory=create_seq2seq_generator,
    )

    container.register_type(_BeamSearchSeq2SeqGeneratorFactory)

    # Standard Algorithm
    def create_standard_beam_search_algorithm(
        resolver: DependencyResolver, config: None
    ) -> BeamSearchAlgorithm:
        return StandardBeamSearchAlgorithm()

    register_component(
        container,
        BeamSearchAlgorithm,
        STANDARD_BEAM_SEARCH_ALGO,
        config_kls=NoneType,
        factory=create_standard_beam_search_algorithm,
    )
