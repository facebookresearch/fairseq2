# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch.nn import Module

from fairseq2.data.tokenizers import Tokenizer
from fairseq2.generation import Seq2SeqGenerator, SequenceGenerator
from fairseq2.generation.beam_search import (
    BeamSearchAlgorithm,
    BeamSearchSeq2SeqGenerator,
    BeamSearchSequenceGenerator,
)
from fairseq2.models.clm import CausalLM
from fairseq2.models.seq2seq import Seq2SeqModel
from fairseq2.recipe.component import ComponentManager, ComponentNotKnownError
from fairseq2.recipe.config import BeamSearchConfig
from fairseq2.recipe.error import (
    BeamSearchAlgorithmNotKnownError,
    raise_model_type_not_valid_error,
)


@final
class _BeamSearchSequenceGeneratorFactory:
    def __init__(
        self,
        model: Module,
        tokenizer: Tokenizer,
        component_manager: ComponentManager,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._component_manager = component_manager

    def create(self, config: BeamSearchConfig) -> SequenceGenerator:
        try:
            algo = self._component_manager.create_component(
                BeamSearchAlgorithm, config.algo.name, config.algo.config
            )
        except ComponentNotKnownError:
            raise BeamSearchAlgorithmNotKnownError(config.algo.name) from None

        if not isinstance(self._model, CausalLM):
            raise_model_type_not_valid_error(self._model, CausalLM)

        max_gen_len = config.max_gen_len

        if isinstance(max_gen_len, tuple):
            max_gen_len = max_gen_len[1]

        return BeamSearchSequenceGenerator(
            self._model,
            self._tokenizer.vocab_info,
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


@final
class _BeamSearchSeq2SeqGeneratorFactory:
    def __init__(
        self,
        model: Module,
        tokenizer: Tokenizer,
        component_manager: ComponentManager,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._component_manager = component_manager

    def create(self, config: BeamSearchConfig) -> Seq2SeqGenerator:
        try:
            algo = self._component_manager.create_component(
                BeamSearchAlgorithm, config.algo.name, config.algo.config
            )
        except ComponentNotKnownError:
            raise BeamSearchAlgorithmNotKnownError(config.algo.name) from None

        if not isinstance(self._model, Seq2SeqModel):
            raise_model_type_not_valid_error(self._model, Seq2SeqModel)

        max_gen_len = config.max_gen_len

        if isinstance(max_gen_len, int):
            max_gen_len = (0.0, max_gen_len)

        return BeamSearchSeq2SeqGenerator(
            self._model,
            self._tokenizer.vocab_info,
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
