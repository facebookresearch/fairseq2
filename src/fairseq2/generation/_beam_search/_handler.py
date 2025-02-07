# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, final

from typing_extensions import override

from fairseq2.generation._beam_search._algo import (
    STANDARD_BEAM_SEARCH_ALGO,
    BeamSearchAlgorithmHandler,
    UnknownBeamSearchAlgorithmError,
)
from fairseq2.generation._beam_search._generator import (
    BeamSearchSeq2SeqGenerator,
    BeamSearchSequenceGenerator,
)
from fairseq2.generation._generator import Seq2SeqGenerator, SequenceGenerator
from fairseq2.generation._handler import (
    Seq2SeqGeneratorHandler,
    SequenceGeneratorHandler,
)
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.registry import Provider
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate

BEAM_SEARCH_GENERATOR: Final = "beam_search"


@dataclass(kw_only=True)
class BeamSearchConfig:
    algorithm: BeamSearchAlgorithmSection = field(
        default_factory=lambda: BeamSearchAlgorithmSection()
    )
    """The beam search algorithm."""

    beam_size: int = 5
    """The beam size."""

    min_gen_len: int = 1
    """The minimum generation length."""

    max_gen_len: int | tuple[int, int] = 2048
    """The maximum generation length."""

    max_seq_len: int | None = None
    """The maximum sequence length including prompt."""

    echo_prompt: bool = False
    """If ``True``, returns generated sequences with prompts appended."""

    normalize_scores: bool = True
    """If ``True``, normalizes scores by lengths of generated sequences."""

    temperature: float = 1.0
    """The logit temperature."""

    unk_penalty: float = 0.0
    """The UNK symbol penalty."""

    len_penalty: float = 1.0
    """The length penalty."""

    prefill_chunk_size: int | None = 512
    """The prefill will be performed incrementally by chunks of this size."""

    decode_capacity_increment: int | None = 16
    """The sequence length capacity will be incremented by multiplies of this value."""


@dataclass(kw_only=True)
class BeamSearchAlgorithmSection:
    name: str = STANDARD_BEAM_SEARCH_ALGO

    config: object = None


@final
class BeamSearchSequenceGeneratorHandler(SequenceGeneratorHandler):
    _algorithm_handlers: Provider[BeamSearchAlgorithmHandler]

    def __init__(
        self, algorithm_handlers: Provider[BeamSearchAlgorithmHandler]
    ) -> None:
        self._algorithm_handlers = algorithm_handlers

    @override
    def create(self, model: DecoderModel, config: object) -> SequenceGenerator:
        config = structure(config, BeamSearchConfig)

        validate(config)

        algorithm_section = config.algorithm

        try:
            algorithm_handler = self._algorithm_handlers.get(algorithm_section.name)
        except LookupError:
            raise UnknownBeamSearchAlgorithmError(algorithm_section.name) from None

        algorithm = algorithm_handler.create(algorithm_section.config)

        if isinstance(config.max_gen_len, int):
            max_gen_len = config.max_gen_len
        else:
            if config.max_gen_len[0] != 1:
                raise ValueError("`max_gen_len` must be an integer.")

            max_gen_len = config.max_gen_len[1]

        return BeamSearchSequenceGenerator(
            model,
            algorithm=algorithm,
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

    @property
    @override
    def config_kls(self) -> type[object]:
        return BeamSearchConfig


@final
class BeamSearchSeq2SeqGeneratorHandler(Seq2SeqGeneratorHandler):
    _algorithm_handlers: Provider[BeamSearchAlgorithmHandler]

    def __init__(
        self, algorithm_handlers: Provider[BeamSearchAlgorithmHandler]
    ) -> None:
        self._algorithm_handlers = algorithm_handlers

    @override
    def create(self, model: EncoderDecoderModel, config: object) -> Seq2SeqGenerator:
        config = structure(config, BeamSearchConfig)

        validate(config)

        algorithm_section = config.algorithm

        try:
            algorithm_handler = self._algorithm_handlers.get(algorithm_section.name)
        except LookupError:
            raise UnknownBeamSearchAlgorithmError(algorithm_section.name) from None

        algorithm = algorithm_handler.create(algorithm_section.config)

        max_gen_len = config.max_gen_len

        if isinstance(max_gen_len, int):
            max_gen_len = (1, max_gen_len)

        return BeamSearchSeq2SeqGenerator(
            model,
            algorithm=algorithm,
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

    @property
    @override
    def config_kls(self) -> type[object]:
        return BeamSearchConfig
