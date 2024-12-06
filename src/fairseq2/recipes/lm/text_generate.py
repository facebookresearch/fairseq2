# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO, final

import torch
from typing_extensions import override

from fairseq2.assets import AssetNotFoundError, default_asset_store
from fairseq2.checkpoint import CheckpointModelMetadataProvider
from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.text import TextTokenDecoder, TextTokenizer, load_text_tokenizer
from fairseq2.datasets import StaticBatching
from fairseq2.datasets.instruction import (
    GenericInstructionDataset,
    load_instruction_dataset,
)
from fairseq2.gang import Gang
from fairseq2.generation import SamplingConfig, SequenceGenerator, create_seq_generator
from fairseq2.logging import get_log_writer
from fairseq2.models import load_model
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.sequence import SequenceBatch
from fairseq2.recipes.common_metrics import SequenceGenerationMetricBag
from fairseq2.recipes.generator import AbstractGeneratorUnit, Generator
from fairseq2.recipes.utils.asset import (
    AssetReference,
    asset_as_path,
    retrieve_asset_card,
)
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.utils.setup import broadcast_model, setup_gangs
from fairseq2.typing import CPU, META, DataClass, DataType
from fairseq2.utils.profiler import Stopwatch
from fairseq2.utils.rng import manual_seed

log = get_log_writer(__name__)


@dataclass(kw_only=True)
class TextGenerateConfig:
    """Holds the configuration of a text generation task."""

    # Data
    dataset: AssetReference = "foo"  # TODO: change!
    """The name, path, or path to the asset card of the instruction dataset."""

    split: str = "default"
    """The name of the data split."""

    max_seq_len: int = 8192
    """The maximum sequence length."""

    batch_size: int = 1
    """The number of prompts per batch."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    # Model
    model: AssetReference = "llama3_8b_instruct"
    """The name of the model to generate with."""

    checkpoint_dir: Path | None = None
    """The checkpoint directory containing models saved by :class:`FileCheckpointManager`."""

    dtype: DataType = torch.bfloat16
    """The data type of the model."""

    amp: bool = False
    """If ``True``, runs evaluation with ``torch.amp``."""

    tensor_parallel_size: int = 1
    """The size of tensor parallelism."""

    # Generation
    generator: str = "sampling"
    """The sequence generator."""

    generator_config: DataClass | None = field(default_factory=lambda: SamplingConfig())
    """The configuration of the sequence generator."""

    # Misc
    seed: int = 2
    """The random number generator seed to use."""


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


@text_generate_preset("llama3_1_8b_instruct")
def _llama3_1_8b_instruct() -> TextGenerateConfig:
    config = _llama3_8b_instruct()

    config.model = "llama3_1_8b_instruct"

    return config


@text_generate_preset("llama3_1_70b_instruct")
def _llama3_1_70b_instruct() -> TextGenerateConfig:
    config = _llama3_70b_instruct()

    config.model = "llama3_1_70b_instruct"

    return config


@torch.inference_mode()
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

    model_card = retrieve_asset_card(config.model)

    # Load the tokenizer.
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
        dataset_path = asset_as_path(config.dataset)

        dataset = GenericInstructionDataset.from_path(dataset_path)

    seed = config.seed

    # Load the model
    manual_seed(seed, CPU, root_gang.device)

    seed += 1

    log.info("Loading {} model on data parallel rank 0 (per shard).", model_card.name)

    if dp_gang.rank == 0:
        init_device = dp_gang.device
    else:
        init_device = META

    try:
        model = load_model(
            model_card, gangs=gangs, device=init_device, dtype=config.dtype
        )
    except ValueError as ex:
        raise ValueError(
            "The model cannot be initialized. See nested exception for details."
        ) from ex

    if not isinstance(model, DecoderModel):
        raise ValueError(
            f"The model must be of type `{DecoderModel}`, but is of type `{type(model)}` instead."
        )

    root_gang.barrier()

    log.info("Model loaded on data parallel rank 0.")

    # Distribute the model to all processes in the gang.
    if dp_gang.size != 1:
        broadcast_model(model, dp_gang, log)

    log_model(model, log)

    # Initialize the sequence generator.
    try:
        generator = create_seq_generator(
            config.generator, model, config.generator_config
        )
    except ValueError as ex:
        raise ValueError(
            "The sequence generator cannot be created. See nested exception for details."
        ) from ex

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

    try:
        data_reader = dataset.create_prompt_reader(
            config.split,
            tokenizer,
            dp_gang,
            config.max_seq_len,
            batching=StaticBatching(config.batch_size),
            sync_mode="until_last",
            num_prefetch=config.num_prefetch,
            seed=seed,
        )
    except ValueError as ex:
        raise ValueError(
            "The data reader cannot be initialized. See nested exception for details."
        ) from ex

    seed += 1

    # Initialize the generator.
    return Generator[SequenceBatch](
        unit=unit,
        data_reader=data_reader,
        root_gang=root_gang,
        dp_gang=dp_gang,
        tp_gang=tp_gang,
        dtype=config.dtype,
        amp=config.amp,
        metrics_dir=output_dir.joinpath("metrics"),
        seed=seed,
        wall_watch=wall_watch,
    )


@final
class TextGenerateUnit(AbstractGeneratorUnit[SequenceBatch]):
    """Represents a text generation unit."""

    _generator: SequenceGenerator
    _text_decoder: TextTokenDecoder
    _text_output_stream: TextIO | None
    _json_output_stream: TextIO | None
    _metric_bag: SequenceGenerationMetricBag

    def __init__(
        self,
        generator: SequenceGenerator,
        tokenizer: TextTokenizer,
        gang: Gang,
        text_output_stream: TextIO | None,
        json_output_stream: TextIO | None,
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
            raise ValueError("`batch.example` must contain a 'prompt' item.") from None

        ids = batch.example["id"]

        output = self._generator(batch.seqs, batch.padding_mask)

        self._metric_bag.update_batch_metrics(output)

        # Check if we are in the first tensor parallel group.
        if not self._text_output_stream and not self._json_output_stream:
            return

        for id_, prompt, hypotheses in zip(ids, prompts, output.hypotheses):
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
                if id_ is not None:
                    stream.write("<<<<< ID >>>>>")
                    stream.write("\n")
                    stream.write(f"{id_}")
                    stream.write("\n\n")

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
                    "id": id_,
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
