# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.generation import Seq2SeqGenerator, SequenceGenerator
from fairseq2.generation.sampling import Sampler, TopKSampler, TopPSampler
from fairseq2.recipe.component import register_component
from fairseq2.recipe.config import (
    SAMPLING_GENERATOR,
    TOP_K_SAMPLER,
    TOP_P_SAMPLER,
    SamplingConfig,
    TopKSamplerConfig,
    TopPSamplerConfig,
)
from fairseq2.recipe.internal.sampling import (
    _SamplingSeq2SeqGeneratorFactory,
    _SamplingSequenceGeneratorFactory,
)
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


def _register_sampling(container: DependencyContainer) -> None:
    # Sequence
    def create_seq_generator(
        resolver: DependencyResolver, config: SamplingConfig
    ) -> SequenceGenerator:
        gen_factory = resolver.resolve(_SamplingSequenceGeneratorFactory)

        return gen_factory.create(config)

    register_component(
        container,
        SequenceGenerator,
        SAMPLING_GENERATOR,
        config_kls=SamplingConfig,
        factory=create_seq_generator,
    )

    container.register_type(_SamplingSequenceGeneratorFactory)

    # Seq2Seq
    def create_seq2seq_generator(
        resolver: DependencyResolver, config: SamplingConfig
    ) -> Seq2SeqGenerator:
        gen_factory = resolver.resolve(_SamplingSeq2SeqGeneratorFactory)

        return gen_factory.create(config)

    register_component(
        container,
        Seq2SeqGenerator,
        SAMPLING_GENERATOR,
        config_kls=SamplingConfig,
        factory=create_seq2seq_generator,
    )

    container.register_type(_SamplingSeq2SeqGeneratorFactory)

    # Top-P
    def create_top_p_sampler(
        resolver: DependencyResolver, config: TopPSamplerConfig
    ) -> Sampler:
        return TopPSampler(p=config.p)

    register_component(
        container,
        Sampler,
        TOP_P_SAMPLER,
        config_kls=TopPSamplerConfig,
        factory=create_top_p_sampler,
    )

    # Top-K
    def create_top_k_sampler(
        resolver: DependencyResolver, config: TopKSamplerConfig
    ) -> Sampler:
        return TopKSampler(k=config.k)

    register_component(
        container,
        Sampler,
        TOP_K_SAMPLER,
        config_kls=TopKSamplerConfig,
        factory=create_top_k_sampler,
    )
