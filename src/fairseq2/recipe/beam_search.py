# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.data.tokenizers import Tokenizer
from fairseq2.generation import Seq2SeqGenerator, SequenceGenerator
from fairseq2.generation.beam_search import (
    BeamSearchAlgorithm,
    BeamSearchSeq2SeqGenerator,
    BeamSearchSequenceGenerator,
    StandardBeamSearchAlgorithm,
)
from fairseq2.model.context import ModelContext
from fairseq2.models.clm import CausalLM
from fairseq2.models.seq2seq import Seq2SeqModel
from fairseq2.recipe.component import ComponentManager, UnknownComponentError
from fairseq2.recipe.config import BeamSearchConfig
from fairseq2.recipe.error import UnknownBeamSearchAlgorithmError
from fairseq2.runtime.dependency import DependencyResolver


def _create_beam_search_seq_generator(
    resolver: DependencyResolver, config: BeamSearchConfig
) -> SequenceGenerator:
    model_context = resolver.resolve(ModelContext)

    tokenizer = resolver.resolve(Tokenizer)

    component_manager = resolver.resolve(ComponentManager)

    try:
        algo = component_manager.create_component(
            BeamSearchAlgorithm, config.algo.name, config.algo.config
        )
    except UnknownComponentError:
        raise UnknownBeamSearchAlgorithmError(config.algo.name) from None

    module = model_context.base_module
    if not isinstance(module, CausalLM):
        raise TypeError(
            f"`model.base_module` is expected to be of type `{CausalLM}`, but is of type `{type(module)}` instead."
        )

    max_gen_len = config.max_gen_len

    if isinstance(max_gen_len, tuple):
        max_gen_len = max_gen_len[1]

    return BeamSearchSequenceGenerator(
        module,
        tokenizer.vocab_info,
        algo,
        beam_size=config.beam_size,
        min_gen_len=config.min_gen_len,
        max_gen_len=max_gen_len,
        max_seq_len=config.max_seq_len,
        echo_prompt=config.echo_prompt,
        normalize_scores=config.normalize_scores,
        temperature=config.temperature,
        unk_penalty=config.unk_penalty,
        len_penalty=config.len_penalty,
        prefill_chunk_size=config.prefill_chunk_size,
        decode_capacity_increment=config.decode_capacity_increment,
    )


def _create_beam_search_seq2seq_generator(
    resolver: DependencyResolver, config: BeamSearchConfig
) -> Seq2SeqGenerator:
    model_context = resolver.resolve(ModelContext)

    tokenizer = resolver.resolve(Tokenizer, key="target")

    component_manager = resolver.resolve(ComponentManager)

    try:
        algo = component_manager.create_component(
            BeamSearchAlgorithm, config.algo.name, config.algo.config
        )
    except UnknownComponentError:
        raise UnknownBeamSearchAlgorithmError(config.algo.name) from None

    module = model_context.base_module
    if not isinstance(module, Seq2SeqModel):
        raise TypeError(
            f"`model.base_module` is expected to be of type `{Seq2SeqModel}`, but is of type `{type(module)}` instead."
        )

    max_gen_len = config.max_gen_len

    if isinstance(max_gen_len, int):
        max_gen_len = (0.0, max_gen_len)

    return BeamSearchSeq2SeqGenerator(
        module,
        tokenizer.vocab_info,
        algo,
        beam_size=config.beam_size,
        min_gen_len=config.min_gen_len,
        max_gen_len=max_gen_len,
        max_seq_len=config.max_seq_len,
        echo_prompt=config.echo_prompt,
        normalize_scores=config.normalize_scores,
        temperature=config.temperature,
        unk_penalty=config.unk_penalty,
        len_penalty=config.len_penalty,
        prefill_chunk_size=config.prefill_chunk_size,
        decode_capacity_increment=config.decode_capacity_increment,
    )


def _create_standard_beam_search_algorithm(
    resolver: DependencyResolver, config: None
) -> BeamSearchAlgorithm:
    return StandardBeamSearchAlgorithm()
