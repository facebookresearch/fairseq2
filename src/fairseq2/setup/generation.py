# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.generation import (
    BEAM_SEARCH_GENERATOR,
    SAMPLING_GENERATOR,
    STANDARD_BEAM_SEARCH_ALGO,
    TOP_K_SAMPLER,
    TOP_P_SAMPLER,
    BeamSearchAlgorithmHandler,
    BeamSearchSeq2SeqGeneratorHandler,
    BeamSearchSequenceGeneratorHandler,
    SamplerHandler,
    SamplingSeq2SeqGeneratorHandler,
    SamplingSequenceGeneratorHandler,
    Seq2SeqGeneratorHandler,
    SequenceGeneratorHandler,
    StandardBeamSearchAlgorithmHandler,
    TopKSamplerHandler,
    TopPSamplerHandler,
)


def _register_seq_generators(context: RuntimeContext) -> None:
    registry = context.get_registry(SequenceGeneratorHandler)

    handler: SequenceGeneratorHandler

    # Sampling
    sampler_handlers = context.get_registry(SamplerHandler)

    handler = SamplingSequenceGeneratorHandler(sampler_handlers)

    registry.register(SAMPLING_GENERATOR, handler)

    # Beam Search
    algorithm_handlers = context.get_registry(BeamSearchAlgorithmHandler)

    handler = BeamSearchSequenceGeneratorHandler(algorithm_handlers)

    registry.register(BEAM_SEARCH_GENERATOR, handler)


def _register_seq2seq_generators(context: RuntimeContext) -> None:
    registry = context.get_registry(Seq2SeqGeneratorHandler)

    handler: Seq2SeqGeneratorHandler

    # Sampling
    sampler_handlers = context.get_registry(SamplerHandler)

    handler = SamplingSeq2SeqGeneratorHandler(sampler_handlers)

    registry.register(SAMPLING_GENERATOR, handler)

    # Beam Search
    algorithm_handlers = context.get_registry(BeamSearchAlgorithmHandler)

    handler = BeamSearchSeq2SeqGeneratorHandler(algorithm_handlers)

    registry.register(BEAM_SEARCH_GENERATOR, handler)


def _register_samplers(context: RuntimeContext) -> None:
    registry = context.get_registry(SamplerHandler)

    registry.register(TOP_P_SAMPLER, TopPSamplerHandler())
    registry.register(TOP_K_SAMPLER, TopKSamplerHandler())


def _register_beam_search_algorithms(context: RuntimeContext) -> None:
    registry = context.get_registry(BeamSearchAlgorithmHandler)

    registry.register(STANDARD_BEAM_SEARCH_ALGO, StandardBeamSearchAlgorithmHandler())
