# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, TextIO, Union, final

import torch

from fairseq2.assets import AssetNotFoundError, default_asset_store
from fairseq2.assets.utils import retrieve_asset_card
from fairseq2.checkpoint import CheckpointModelMetadataProvider
from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.text import TextTokenDecoder, TextTokenizer, load_text_tokenizer
from fairseq2.datasets import StaticBatching
from fairseq2.datasets.instruction import (
    GenericInstructionDataset,
    load_instruction_dataset,
)
from fairseq2.gang import Gang
from fairseq2.generation import (
    BeamSearchSequenceGenerator,
    Sampler,
    SamplingSequenceGenerator,
    SequenceGenerator,
    StandardBeamSearchAlgorithm,
    TopKSampler,
    TopPSampler,
)
from fairseq2.logging import get_log_writer
from fairseq2.models import load_model
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.sequence import SequenceBatch
from fairseq2.recipes.common_metrics import SequenceGenerationMetricBag
from fairseq2.recipes.generator import AbstractGeneratorUnit, Generator
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.utils.setup import broadcast_model, setup_gangs
from fairseq2.typing import META, DataType, override
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


@dataclass
class TextGenerateConfig:
    """Holds the configuration of a text generation task."""

    # Data
    dataset: Union[str, Path] = "oa2_gsm8k_safety"  # TODO: change!
    """The name, path, or path to the asset card of the instruction dataset."""

    max_seq_len: int = 8192
    """The maximum sequence length."""

    batch_size: int = 1
    """The input batch size."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    # Model
    model: Union[str, Path] = "llama3_8b_instruct"
    """The name of the model to generate with."""

    checkpoint_dir: Optional[Path] = None
    """The checkpoint directory containing models saved by :class:`FileCheckpointManager`."""

    dtype: DataType = torch.bfloat16
    """The data type of the model."""

    tensor_parallel_size: int = 1
    """The size of tensor parallelism."""

    # Generation
    mode: Literal["sampling", "beam_search"] = "sampling"
    """The mode of sequence generation."""

    sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig())
    """The configuration for sampling-based sequence generation."""

    beam_search: BeamSearchConfig = field(default_factory=lambda: BeamSearchConfig())
    """The configuration for beam search-based sequence generation."""

    # Misc
    seed: int = 2
    """The random number generator seed to use."""


@dataclass
class SamplingConfig:
    """Holds the configuration for sampling-based sequence generation.

    See :class:`SamplingSequenceGenerator` for more info.
    """

    sampler: Literal["top-p", "top-k"] = "top-p"
    """The sampling algorithm."""

    top_p: float = 0.9
    """The cumulative probability threshold for top-p sampling."""

    top_k = 10
    """The number of top candidates to select from for top-k sampling."""

    min_gen_len: int = 1
    """The minimum generation length."""

    max_gen_len: int = 2048
    """The maximum generation length."""

    max_seq_len: Optional[int] = None
    """The maximum sequence length including prompt."""

    echo_prompt: bool = False
    """If ``True``, returns generated sequences with prompts appended."""

    compute_scores: bool = False
    """If ``True``, computes scores of generated sequences."""

    normalize_scores: bool = True
    """If ``True``, normalizes scores by lengths of generated sequences."""

    temperature: float = 0.6
    """The logit temperature."""

    unk_penalty: float = 0.0
    """The UNK symbol penalty."""

    len_penalty: float = 1.0
    """The length penalty."""

    prefill_chunk_size: Optional[int] = 512
    """The prefill will be performed incrementally by chunks of this size."""

    decode_capacity_increment: Optional[int] = 16
    """The sequence length capacity will be incremented by multiplies of this value."""


@dataclass
class BeamSearchConfig:
    """Holds the configuration for beam search-based sequence generation.

    See :class:`BeamSearchSequenceGenerator` for more info.
    """

    algorithm: Literal["standard"] = "standard"
    """The beam search algorithm."""

    beam_size: int = 5
    """The beam size."""

    min_gen_len: int = 1
    """The minimum generation length."""

    max_gen_len: int = 512
    """The maximum generation length."""

    max_seq_len: Optional[int] = None
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

    prefill_chunk_size: Optional[int] = 512
    """The prefill will be performed incrementally by chunks of this size."""

    decode_capacity_increment: Optional[int] = 16
    """The sequence length capacity will be incremented by multiplies of this value."""


text_generate_presets = ConfigRegistry[TextGenerateConfig]()

text_generate_preset = text_generate_presets.decorator


@text_generate_preset("llama2_7b_chat")
def _llama2_7b_chat() -> TextGenerateConfig:
    config = _llama3_8b_instruct()

    config.model = "llama2_7b_chat"

    return config


@text_generate_preset("llama2_70b_chat")
def _llama2_70b_chat() -> TextGenerateConfig:
    config = _llama2_7b_chat()

    config.model = "llama2_70b_chat"
    config.tensor_parallel_size = 8

    return config


@text_generate_preset("llama3_8b_instruct")
def _llama3_8b_instruct() -> TextGenerateConfig:
    return TextGenerateConfig()


@text_generate_preset("llama3_70b_instruct")
def _llama3_70b_instruct() -> TextGenerateConfig:
    config = _llama3_8b_instruct()

    config.model = "llama3_70b_instruct"
    config.tensor_parallel_size = 8

    return config


def load_text_generator(
    config: TextGenerateConfig, output_dir: Path
) -> Generator[SequenceBatch]:
    """Load a :class:`Generator` for text generation."""
    wall_watch = Stopwatch(start=True)

    if config.checkpoint_dir is not None:
        default_asset_store.metadata_providers.append(
            CheckpointModelMetadataProvider(config.checkpoint_dir)
        )

    root_gang, gangs = setup_gangs(log, tp_size=config.tensor_parallel_size)

    dp_gang = gangs["dp"]  # data
    tp_gang = gangs["tp"]  # tensor

    seed = config.seed

    # Load the tokenizer.
    model_card = retrieve_asset_card(config.model)

    log.info("Loading {} tokenizer.", model_card.name)

    tokenizer = load_text_tokenizer(model_card)

    log.info("Tokenizer loaded.")

    # Load the dataset.
    try:
        dataset_card = retrieve_asset_card(config.dataset)
    except AssetNotFoundError:
        dataset_card = None

    if dataset_card is not None:
        log.info("Loading {} instruction dataset.", dataset_card.name)

        dataset = load_instruction_dataset(dataset_card)

        log.info("Dataset loaded.")
    else:
        try:
            path = Path(config.dataset)
        except ValueError:
            raise AssetNotFoundError(
                config.dataset, f"An asset with the name '{config.dataset}' cannot be found."  # type: ignore[arg-type]
            )

        dataset = GenericInstructionDataset.from_path(path)

    # Load the model.
    log.info("Loading {} model on data parallel rank 0 (per shard).", model_card.name)

    if dp_gang.rank == 0:
        init_device = dp_gang.device
    else:
        init_device = META

    model = load_model(model_card, gangs=gangs, device=init_device, dtype=config.dtype)

    root_gang.barrier()

    log.info("Model loaded on data parallel rank 0.")

    if not isinstance(model, DecoderModel):
        raise ValueError("`config.model` must specify a decoder model.")

    # Distribute the model to all processes in the gang.
    if dp_gang.size != 1:
        broadcast_model(model, dp_gang, log)

    log_model(model, log)

    # Initialize the sequence generator.
    generator = _create_sequence_generator(
        model, config.mode, config.beam_search, config.sampling
    )

    # Initialize the generator unit.
    if tp_gang.rank == 0:
        text_output_file = output_dir.joinpath(f"output/rank_{dp_gang.rank}.txt")
        json_output_file = output_dir.joinpath(f"output/rank_{dp_gang.rank}.jsonl")

        try:
            text_output_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as ex:
            raise RuntimeError(
                f"The output directory '{text_output_file.parent}' cannot be created. See nested exception for details."
            ) from ex

        try:
            text_output_fp = text_output_file.open("w")
        except OSError as ex:
            raise RuntimeError(
                f"The output file '{text_output_file}' cannot be created. See nested exception for details."
            ) from ex

        try:
            json_output_fp = json_output_file.open("w")
        except OSError as ex:
            raise RuntimeError(
                f"The output file '{json_output_file}' cannot be created. See nested exception for details."
            ) from ex

    else:
        text_output_fp = None
        json_output_fp = None

    unit = TextGenerateUnit(
        generator,
        tokenizer,
        dp_gang,
        text_output_stream=text_output_fp,
        json_output_stream=json_output_fp,
    )

    data_reader = dataset.create_prompt_reader(
        tokenizer,
        dp_gang,
        config.max_seq_len,
        batching=StaticBatching(config.batch_size),
        num_prefetch=config.num_prefetch,
        seed=seed,
    )

    seed += 1

    # Initialize the generator.
    return Generator[SequenceBatch](
        unit=unit,
        data_reader=data_reader,
        root_gang=root_gang,
        dp_gang=dp_gang,
        tp_gang=tp_gang,
        metrics_dir=output_dir.joinpath("metrics"),
        seed=seed,
        wall_watch=wall_watch,
    )


@final
class TextGenerateUnit(AbstractGeneratorUnit[SequenceBatch]):
    """Represents a text generation unit."""

    _generator: SequenceGenerator
    _text_decoder: TextTokenDecoder
    _text_output_stream: Optional[TextIO]
    _json_output_stream: Optional[TextIO]
    _metric_bag: SequenceGenerationMetricBag

    def __init__(
        self,
        generator: SequenceGenerator,
        tokenizer: TextTokenizer,
        gang: Gang,
        text_output_stream: Optional[TextIO],
        json_output_stream: Optional[TextIO],
    ) -> None:
        super().__init__(generator.model)

        self._generator = generator

        self._text_decoder = tokenizer.create_decoder()

        self._text_output_stream = text_output_stream
        self._json_output_stream = json_output_stream

        self._metric_bag = SequenceGenerationMetricBag(gang)

    @override
    def __call__(self, batch: SequenceBatch) -> None:
        if batch.example is None:
            raise ValueError("`batch.example` must not be `None`.")

        try:
            prompts = batch.example["prompt"]
        except KeyError:
            raise ValueError("`batch.example` must contain a 'prompt' item.")

        output = self._generator(batch.seqs, batch.padding_mask)

        self._metric_bag.update_batch_metrics(output)

        # Check if we are in the first tensor parallel group.
        if not self._text_output_stream and not self._json_output_stream:
            return

        for prompt, hypotheses in zip(prompts, output.hypotheses):
            if len(hypotheses) == 0:
                raise RuntimeError(
                    "The sequence generator returned no hypothesis. Please file a bug report."
                )

            hypothesis = hypotheses[0]

            seq = hypothesis.seq

            response = self._text_decoder(seq)

            token_indices = seq.tolist()

            if hypothesis.score is None:
                score = None
            else:
                score = float(hypothesis.score)

            if hypothesis.step_scores is None:
                step_scores = None
            else:
                step_scores = hypothesis.step_scores.tolist()

            # Dump as text.
            if stream := self._text_output_stream:
                stream.write("<<<<< PROMPT >>>>>")
                stream.write("\n")
                stream.write(prompt)

                stream.write("\n\n")
                stream.write("<<<<< RESPONSE >>>>>")
                stream.write("\n")
                stream.write(response)

                stream.write("\n\n")
                stream.write("<<<<< TOKEN INDICES >>>>>")
                stream.write("\n")
                stream.write(", ".join(f"{t}" for t in token_indices))

                if score is not None:
                    stream.write("\n\n")
                    stream.write("<<<<< SCORE >>>>>")
                    stream.write("\n")

                    stream.write(f"{score:.8f}")

                if step_scores is not None:
                    stream.write("\n\n")
                    stream.write("<<<<< STEP SCORES >>>>>")
                    stream.write("\n")

                    stream.write(", ".join(f"{s:.8f}" for s in step_scores))

                stream.write("\n\n\n============================\n\n\n")

            # Dump as JSON.
            if stream := self._json_output_stream:
                json_output = {
                    "prompt": prompt,
                    "response": response,
                    "token_indices": token_indices,
                    "score": score,
                    "step_scores": step_scores,
                }

                json.dump(json_output, stream, indent=None)

                stream.write("\n")

        if stream := self._text_output_stream:
            stream.flush()

        if stream := self._json_output_stream:
            stream.flush()

    @property
    @override
    def metric_bag(self) -> SequenceGenerationMetricBag:
        return self._metric_bag


def _create_sequence_generator(
    model: DecoderModel,
    mode: str,
    beam_search_config: BeamSearchConfig,
    sampling_config: SamplingConfig,
) -> SequenceGenerator:
    if mode == "sampling":
        return _create_sampling_generator(model, sampling_config)

    if mode == "beam_search":
        return _create_beam_search_generator(model, beam_search_config)

    raise ValueError(
        f"`config.mode` must be 'sampling' or 'beam_search', but is '{mode}' instead."
    )


def _create_sampling_generator(
    model: DecoderModel, config: SamplingConfig
) -> SamplingSequenceGenerator:
    sampler: Sampler

    if config.sampler == "top-p":
        sampler = TopPSampler(config.top_p)
    elif config.sampler == "top-k":
        sampler = TopKSampler(config.top_k)
    else:
        raise ValueError(
            f"`config.sampling.sampler` must be 'top-p' or 'top-k', but is '{config.sampler}' instead."
        )

    return SamplingSequenceGenerator(
        model,
        sampler,
        min_gen_len=config.min_gen_len,
        max_gen_len=config.max_gen_len,
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


def _create_beam_search_generator(
    model: DecoderModel, config: BeamSearchConfig
) -> BeamSearchSequenceGenerator:
    if config.algorithm == "standard":
        algorithm = StandardBeamSearchAlgorithm()
    else:
        raise ValueError(
            f"`config.beam_search.algorithm` must be 'standard', but is '{config.algorithm}' instead."
        )

    return BeamSearchSequenceGenerator(
        model,
        algorithm=algorithm,
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
