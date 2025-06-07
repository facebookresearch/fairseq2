# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.data.tokenizers import Tokenizer
from fairseq2.dependency import DependencyResolver
from fairseq2.generation import Seq2SeqGenerator, SequenceGenerator
from fairseq2.generation.beam_search import (
    BeamSearchAlgorithm,
    BeamSearchSeq2SeqGenerator,
    BeamSearchSequenceGenerator,
    StandardBeamSearchAlgorithm,
)
from fairseq2.models.clm import CausalLM
from fairseq2.models.seq2seq import Seq2SeqModel
from fairseq2.recipe.component import resolve_component
from fairseq2.recipe.config import BeamSearchConfig
from fairseq2.recipe.model import Model
from fairseq2.utils.structured import StructureError


def create_beam_search_seq_generator(
    resolver: DependencyResolver, config: BeamSearchConfig
) -> SequenceGenerator:
    model = resolver.resolve(Model)

    tokenizer = resolver.resolve(Tokenizer)

    try:
        algo = resolve_component(
            resolver, BeamSearchAlgorithm, config.algo.name, config.algo.config
        )
    except StructureError as ex:
        raise StructureError(
            f"The '{config.algo.name}' beam search algorithm configuration cannot be parsed. See the nested exception for details."
        ) from ex

    base_module = model.base_module
    if not isinstance(base_module, CausalLM):
        raise TypeError(
            f"`model.base_module` is expected to be of type `{CausalLM}`, but is of type `{type(base_module)}` instead."
        )

    if isinstance(config.max_gen_len, int):
        max_gen_len = config.max_gen_len
    else:
        if config.max_gen_len[0] != 1:
            raise ValueError("`max_gen_len` must be an integer.")

        max_gen_len = config.max_gen_len[1]

    return BeamSearchSequenceGenerator(
        base_module,
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


def create_beam_search_seq2seq_generator(
    resolver: DependencyResolver, config: BeamSearchConfig
) -> Seq2SeqGenerator:
    model = resolver.resolve(Model)

    tokenizer = resolver.resolve(Tokenizer, key="target")

    try:
        algo = resolve_component(
            resolver, BeamSearchAlgorithm, config.algo.name, config.algo.config
        )
    except StructureError as ex:
        raise StructureError(
            f"The '{config.algo.name}' beam search algorithm configuration cannot be parsed. See the nested exception for details."
        ) from ex

    base_module = model.base_module
    if not isinstance(base_module, Seq2SeqModel):
        raise TypeError(
            f"`model.base_module` is expected to be of type `{Seq2SeqModel}`, but is of type `{type(base_module)}` instead."
        )

    max_gen_len = config.max_gen_len

    if isinstance(max_gen_len, int):
        max_gen_len = (1, max_gen_len)

    return BeamSearchSeq2SeqGenerator(
        base_module,
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


def create_standard_beam_search_algorithm(
    resolver: DependencyResolver, config: None
) -> BeamSearchAlgorithm:
    return StandardBeamSearchAlgorithm()
