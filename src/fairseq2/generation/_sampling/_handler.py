# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, final

from typing_extensions import override

from fairseq2.generation._generator import Seq2SeqGenerator, SequenceGenerator
from fairseq2.generation._handler import (
    Seq2SeqGeneratorHandler,
    SequenceGeneratorHandler,
)
from fairseq2.generation._sampling._generator import (
    SamplingSeq2SeqGenerator,
    SamplingSequenceGenerator,
)
from fairseq2.generation._sampling._sampler import (
    TOP_P_SAMPLER,
    SamplerHandler,
    TopPSamplerConfig,
    UnknownSamplerError,
)
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.registry import Provider
from fairseq2.utils.structured import StructureError, structure
from fairseq2.utils.validation import validate

SAMPLING_GENERATOR: Final = "sampling"


@dataclass(kw_only=True)
class SamplingConfig:
    sampler: SamplerSection = field(default_factory=lambda: SamplerSection())
    """The configuration of the sampler."""

    min_gen_len: int = 1
    """The minimum generation length."""

    max_gen_len: int | tuple[int, int] = 2048
    """The maximum generation length."""

    max_seq_len: int | None = None
    """The maximum sequence length including prompt."""

    echo_prompt: bool = False
    """If ``True``, returns generated sequences with prompts appended."""

    compute_scores: bool = False
    """If ``True``, computes scores of generated sequences."""

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
class SamplerSection:
    name: str = TOP_P_SAMPLER

    config: object = field(default_factory=lambda: TopPSamplerConfig())


@final
class SamplingSequenceGeneratorHandler(SequenceGeneratorHandler):
    _sampler_handlers: Provider[SamplerHandler]

    def __init__(self, sampler_handlers: Provider[SamplerHandler]) -> None:
        self._sampler_handlers = sampler_handlers

    @override
    def create(self, model: DecoderModel, config: object) -> SequenceGenerator:
        config = structure(config, SamplingConfig)

        validate(config)

        sampler_section = config.sampler

        try:
            sampler_handler = self._sampler_handlers.get(sampler_section.name)
        except LookupError:
            raise UnknownSamplerError(sampler_section.name) from None

        try:
            sampler = sampler_handler.create(sampler_section.config)
        except StructureError as ex:
            raise StructureError(
                "`sampler.config` cannot be structured. See the nested exception for details."
            ) from ex

        if isinstance(config.max_gen_len, int):
            max_gen_len = config.max_gen_len
        else:
            if config.max_gen_len[0] != 1:
                raise ValueError("`max_gen_len` must be an integer.")

            max_gen_len = config.max_gen_len[1]

        return SamplingSequenceGenerator(
            model,
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

    @property
    @override
    def config_kls(self) -> type[object]:
        return SamplingConfig


@final
class SamplingSeq2SeqGeneratorHandler(Seq2SeqGeneratorHandler):
    _sampler_handlers: Provider[SamplerHandler]

    def __init__(self, sampler_handlers: Provider[SamplerHandler]) -> None:
        self._sampler_handlers = sampler_handlers

    @override
    def create(self, model: EncoderDecoderModel, config: object) -> Seq2SeqGenerator:
        config = structure(config, SamplingConfig)

        validate(config)

        sampler_section = config.sampler

        try:
            sampler_handler = self._sampler_handlers.get(sampler_section.name)
        except LookupError:
            raise UnknownSamplerError(sampler_section.name) from None

        try:
            sampler = sampler_handler.create(sampler_section.config)
        except StructureError as ex:
            raise StructureError(
                "`sampler.config` cannot be structured. See the nested exception for details."
            ) from ex

        max_gen_len = config.max_gen_len

        if isinstance(max_gen_len, int):
            max_gen_len = (1, max_gen_len)

        return SamplingSeq2SeqGenerator(
            model,
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

    @property
    @override
    def config_kls(self) -> type[object]:
        return SamplingConfig
