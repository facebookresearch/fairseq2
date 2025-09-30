# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from fairseq2.data.tokenizers import Tokenizer
from fairseq2.generation import Seq2SeqGenerator, SequenceGenerator
from fairseq2.generation.sampling import (
    Sampler,
    SamplingSeq2SeqGenerator,
    SamplingSequenceGenerator,
)
from fairseq2.models.clm import CausalLM
from fairseq2.models.seq2seq import Seq2SeqModel
from fairseq2.recipe.component import ComponentManager, ComponentNotKnownError
from fairseq2.recipe.config import SamplingConfig
from fairseq2.recipe.error import SamplerNotKnownError
from fairseq2.recipe.model import RecipeModel


@final
class _SamplingSequenceGeneratorFactory:
    def __init__(
        self,
        model: RecipeModel,
        tokenizer: Tokenizer,
        component_manager: ComponentManager,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._component_manager = component_manager

    def create(self, config: SamplingConfig) -> SequenceGenerator:
        try:
            sampler = self._component_manager.create_component(
                Sampler, config.sampler.name, config.sampler.config
            )
        except ComponentNotKnownError:
            raise SamplerNotKnownError(config.sampler.name) from None

        module = self._model.base_module
        if not isinstance(module, CausalLM):
            raise TypeError(
                f"`model.base_module` is expected to be of type `{CausalLM}`, but is of type `{type(module)}` instead."
            )

        max_gen_len = config.max_gen_len

        if isinstance(max_gen_len, tuple):
            max_gen_len = max_gen_len[1]

        return SamplingSequenceGenerator(
            module,
            self._tokenizer.vocab_info,
            sampler,
            min_gen_len=config.min_gen_len,
            max_gen_len=max_gen_len,
            max_seq_len=config.max_seq_len,
            echo_prompt=config.echo_prompt,
            compute_scores=config.compute_scores,
            normalize_scores=config.normalize_scores,
            temperature=config.temperature,
            unk_penalty=config.unk_penalty,
            len_penalty=config.len_penalty,
            prefill_chunk_size=config.prefill_chunk_size,
            decode_capacity_increment=config.decode_capacity_increment,
        )


@final
class _SamplingSeq2SeqGeneratorFactory:
    def __init__(
        self,
        model: RecipeModel,
        tokenizer: Tokenizer,
        component_manager: ComponentManager,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._component_manager = component_manager

    def create(self, config: SamplingConfig) -> Seq2SeqGenerator:
        try:
            sampler = self._component_manager.create_component(
                Sampler, config.sampler.name, config.sampler.config
            )
        except ComponentNotKnownError:
            raise SamplerNotKnownError(config.sampler.name) from None

        module = self._model.base_module
        if not isinstance(module, Seq2SeqModel):
            raise TypeError(
                f"`model.base_module` is expected to be of type `{Seq2SeqModel}`, but is of type `{type(module)}` instead."
            )

        max_gen_len = config.max_gen_len

        if isinstance(max_gen_len, int):
            max_gen_len = (0.0, max_gen_len)

        return SamplingSeq2SeqGenerator(
            module,
            self._tokenizer.vocab_info,
            sampler,
            min_gen_len=config.min_gen_len,
            max_gen_len=max_gen_len,
            max_seq_len=config.max_seq_len,
            echo_prompt=config.echo_prompt,
            compute_scores=config.compute_scores,
            normalize_scores=config.normalize_scores,
            temperature=config.temperature,
            unk_penalty=config.unk_penalty,
            len_penalty=config.len_penalty,
            prefill_chunk_size=config.prefill_chunk_size,
            decode_capacity_increment=config.decode_capacity_increment,
        )
