# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

from fairseq2.factory_registry import ConfigBoundFactoryRegistry
from fairseq2.generation.beam_search import BeamSearchSeq2SeqGenerator
from fairseq2.generation.beam_search_algorithm import beam_search_factories
from fairseq2.generation.generator import Seq2SeqGenerator
from fairseq2.generation.sampler import TopPSamplerConfig, sampler_factories
from fairseq2.generation.sampling import SamplingSeq2SeqGenerator
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.typing import DataClass

if TYPE_CHECKING:  # compat: remove when Python 3.9 support is dropped.
    generator_factories = ConfigBoundFactoryRegistry[
        [EncoderDecoderModel], Seq2SeqGenerator
    ]()
else:
    generator_factories = ConfigBoundFactoryRegistry()


@dataclass
class BeamSearchConfig:
    """Holds the configuration of a :class:`BeamSearchSeq2SeqGenerator`."""

    algorithm: str = "standard"
    """The beam search algorithm."""

    algorithm_config: Optional[DataClass] = None
    """The configuration of the beam-search algorithm."""

    beam_size: int = 5
    """The beam size."""

    min_gen_len: int = 1
    """The minimum generation length."""

    max_gen_len: Tuple[int, int] = (1, 256)
    """The maximum generation length."""

    max_seq_len: Optional[int] = None
    """The maximum sequence length including prompt."""

    echo_prompt: bool = True
    """If ``True``, returns generated sequences with prompts appended."""

    normalize_scores: bool = True
    """If ``True``, normalizes scores by lengths of generated sequences."""

    temperature: float = 1.0
    """The logit temperature."""

    unk_penalty: float = 0.0
    """The UNK symbol penalty."""

    len_penalty: float = 1.0
    """The length penalty."""

    prefill_chunk_size: Optional[int] = 512
    """The prefill will be performed incrementally by chunks of this size."""

    decode_capacity_increment: Optional[int] = 16
    """The sequence length capacity will be incremented by multiplies of this value."""


def create_beam_search_generator(
    config: BeamSearchConfig, model: EncoderDecoderModel
) -> BeamSearchSeq2SeqGenerator:
    try:
        beam_search_factory = beam_search_factories.get(
            config.algorithm, config.algorithm_config
        )

        beam_search = beam_search_factory()
    except ValueError as ex:
        raise ValueError(
            "The beam-search algorithm cannot be created. See nested exception for details."
        ) from ex

    return BeamSearchSeq2SeqGenerator(
        model,
        algorithm=beam_search,
        beam_size=config.beam_size,
        min_gen_len=config.min_gen_len,
        max_gen_len=config.max_gen_len,
        echo_prompt=config.echo_prompt,
        normalize_scores=config.normalize_scores,
        temperature=config.temperature,
        unk_penalty=config.unk_penalty,
        len_penalty=config.len_penalty,
        prefill_chunk_size=config.prefill_chunk_size,
        decode_capacity_increment=config.decode_capacity_increment,
    )


generator_factories.register(
    "beam_search", create_beam_search_generator, BeamSearchConfig
)


@dataclass
class SamplingConfig:
    """Holds the configuration of a :class:`SamplingSeq2SeqGenerator`."""

    sampler: str = "top-p"
    """The sampler."""

    sampler_config: Optional[DataClass] = field(
        default_factory=lambda: TopPSamplerConfig()
    )
    """The configuration of the sampler."""

    min_gen_len: int = 1
    """The minimum generation length."""

    max_gen_len: Tuple[int, int] = (1, 256)
    """The maximum generation length."""

    max_seq_len: Optional[int] = None
    """The maximum sequence length including prompt."""

    echo_prompt: bool = True
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

    prefill_chunk_size: Optional[int] = 512
    """The prefill will be performed incrementally by chunks of this size."""

    decode_capacity_increment: Optional[int] = 16
    """The sequence length capacity will be incremented by multiplies of this value."""


def create_sampling_generator(
    config: SamplingConfig, model: EncoderDecoderModel
) -> SamplingSeq2SeqGenerator:
    try:
        sampler_factory = sampler_factories.get(config.sampler, config.sampler_config)

        sampler = sampler_factory()
    except ValueError as ex:
        raise ValueError(
            "The sampler cannot be created. See nested exception for details."
        ) from ex

    return SamplingSeq2SeqGenerator(
        model,
        sampler,
        min_gen_len=config.min_gen_len,
        max_gen_len=config.max_gen_len,
        max_seq_len=config.max_seq_len,
        echo_prompt=True,
        compute_scores=config.compute_scores,
        normalize_scores=config.normalize_scores,
        temperature=config.temperature,
        unk_penalty=config.unk_penalty,
        len_penalty=config.len_penalty,
        prefill_chunk_size=config.prefill_chunk_size,
        decode_capacity_increment=config.decode_capacity_increment,
    )


generator_factories.register("sampling", create_sampling_generator, SamplingConfig)
