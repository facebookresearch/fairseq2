# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, TextIO, final

import torch
from typing_extensions import override

from fairseq2.context import RuntimeContext
from fairseq2.data.text.tokenizers import TextTokenDecoder, TextTokenizer
from fairseq2.datasets import StaticBatching, SyncMode
from fairseq2.datasets.instruction import (
    GENERIC_INSTRUCTION_DATASET_FAMILY,
    InstructionDataset,
    InstructionPromptReadOptions,
)
from fairseq2.error import InternalError, SetupError
from fairseq2.gang import Gangs
from fairseq2.generation import SamplingConfig, SequenceGenerator
from fairseq2.logging import log
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.utils.module import remove_parametrizations
from fairseq2.recipes.common import (
    broadcast_model,
    compile_eval_model,
    create_generator,
    create_seq_generator,
    load_dataset,
    load_eval_model,
    load_text_tokenizer,
    register_extra_asset_paths,
    setup_gangs,
)
from fairseq2.recipes.config import (
    DatasetSection,
    GenerateRecipeConfig,
    GeneratorSection,
    SequenceGeneratorSection,
)
from fairseq2.recipes.generator import AbstractGeneratorUnit, Generator
from fairseq2.recipes.metrics import SequenceGenerationMetricBag
from fairseq2.recipes.utils.log import log_model
from fairseq2.typing import CPU
from fairseq2.utils.config import process_config
from fairseq2.utils.file import FileMode
from fairseq2.utils.rng import manual_seed


@dataclass(kw_only=True)
class TextGenerateConfig(GenerateRecipeConfig):
    """Holds the configuration of a text generation task."""

    model: str = "llama3_8b_instruct"

    dataset: TextGenerateDatasetSection = field(
        default_factory=lambda: TextGenerateDatasetSection()
    )

    generator: GeneratorSection = field(
        default_factory=lambda: GeneratorSection(dtype=torch.bfloat16)
    )

    seq_generator: SequenceGeneratorSection = field(
        default_factory=lambda: SequenceGeneratorSection(
            config=SamplingConfig(), batch_size=1
        )
    )


@dataclass(kw_only=True)
class TextGenerateDatasetSection(DatasetSection):
    name: str = "foo"  # TODO: change!

    family: str = GENERIC_INSTRUCTION_DATASET_FAMILY

    path: Path | None = None

    split: str = "default"

    min_seq_len: int = 1
    """The minimum sequence length."""

    max_seq_len: int = 8192
    """The maximum sequence length."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""


def register_text_generate_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(TextGenerateConfig)

    preset = registry.decorator

    @preset("llama2_7b_chat")
    def llama2_7b_chat() -> TextGenerateConfig:
        config = llama3_8b_instruct()

        config.model = "llama2_7b_chat"

        return config

    @preset("llama2_70b_chat")
    def llama2_70b_chat() -> TextGenerateConfig:
        config = llama2_7b_chat()

        config.model = "llama2_70b_chat"
        config.gang.tensor_parallel_size = 8

        return config

    @preset("llama3_8b_instruct")
    def llama3_8b_instruct() -> TextGenerateConfig:
        return TextGenerateConfig()

    @preset("llama3_70b_instruct")
    def llama3_70b_instruct() -> TextGenerateConfig:
        config = llama3_8b_instruct()

        config.model = "llama3_70b_instruct"
        config.gang.tensor_parallel_size = 8

        return config

    @preset("llama3_1_8b_instruct")
    def llama3_1_8b_instruct() -> TextGenerateConfig:
        config = llama3_8b_instruct()

        config.model = "llama3_1_8b_instruct"

        return config

    @preset("llama3_1_70b_instruct")
    def llama3_1_70b_instruct() -> TextGenerateConfig:
        config = llama3_70b_instruct()

        config.model = "llama3_1_70b_instruct"

        return config


@torch.inference_mode()
def load_text_generator(
    context: RuntimeContext, config: TextGenerateConfig, output_dir: Path
) -> Generator[SequenceBatch]:
    register_extra_asset_paths(context, config.assets)

    process_config(context, config)

    gangs = setup_gangs(context, config.gang)

    dataset = load_dataset(InstructionDataset, context, config.dataset, gangs)

    tokenizer = load_text_tokenizer(context, config.model)

    seed = config.seed

    manual_seed(seed, CPU, context.device)

    seed += 1

    model = load_eval_model(
        DecoderModel,
        context,
        config.model,
        gangs,
        config.generator.dtype,
        mixed_precision=config.generator.amp,
    )

    broadcast_model(config.model, model, gangs)

    remove_parametrizations(model)

    log_model(log, model, gangs)

    if config.generator.torch_compile:
        model = compile_eval_model(context, config.model, model)

    # Initialize the unit.
    seq_generator = create_seq_generator(context, config.seq_generator, model)

    if gangs.tp.rank == 0:
        file_system = context.file_system

        rank = gangs.dp.rank

        text_file = output_dir.joinpath(f"output/rank_{rank}.txt")
        json_file = output_dir.joinpath(f"output/rank_{rank}.jsonl")

        try:
            file_system.make_directory(text_file.parent)
        except OSError as ex:
            raise SetupError(
                f"The '{text_file.parent}' output directory cannot be created. See the nested exception for details."
            ) from ex

        try:
            text_fp = file_system.open_text(text_file, mode=FileMode.WRITE)
        except OSError as ex:
            raise SetupError(
                f"The '{text_file}' output file cannot be created. See the nested exception for details."
            ) from ex

        try:
            json_fp = file_system.open_text(json_file, mode=FileMode.WRITE)
        except OSError as ex:
            raise SetupError(
                f"The '{json_file}' output file cannot be created. See the nested exception for details."
            ) from ex
    else:
        text_fp = None
        json_fp = None

    unit = TextGenerateUnit(
        seq_generator,
        tokenizer,
        gangs,
        text_output_stream=text_fp,
        json_output_stream=json_fp,
    )

    batching = StaticBatching(config.seq_generator.batch_size)

    read_options = InstructionPromptReadOptions(
        batching=batching,
        sync_mode=SyncMode.UNTIL_LAST,
        num_prefetch=config.dataset.num_prefetch,
        seed=seed,
    )

    data_reader = dataset.create_prompt_reader(
        config.dataset.split,
        tokenizer,
        gangs.dp,
        config.dataset.min_seq_len,
        config.dataset.max_seq_len,
        read_options,
    )

    seed += 1

    return create_generator(context, config, output_dir, unit, data_reader, gangs, seed)


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
        gangs: Gangs,
        text_output_stream: TextIO | None,
        json_output_stream: TextIO | None,
    ) -> None:
        super().__init__(generator.model)

        self._generator = generator

        self._text_decoder = tokenizer.create_decoder()

        self._text_output_stream = text_output_stream
        self._json_output_stream = json_output_stream

        self._metric_bag = SequenceGenerationMetricBag(gangs.dp)

    @override
    def __call__(self, batch: SequenceBatch) -> None:
        if batch.example is None:
            raise ValueError("`batch.example` must not be `None`.")

        if not isinstance(batch.example, Mapping):
            raise TypeError(
                f"`batch.example` must be of type `{Mapping}`, but is of type `{type(batch.example)}` instead."
            )

        try:
            prompts = batch.example["prompt"]
        except KeyError:
            raise ValueError("`batch.example` must contain a 'prompt' item.") from None

        if not isinstance(prompts, Iterable):
            raise TypeError(
                f"`batch.example['prompt'] must be an iterable of strings, but is of type `{type(prompts)}` instead."
            )

        ids = batch.example["id"]

        output = self._generator(batch.seqs, batch.padding_mask)

        self._metric_bag.update_batch_metrics(output)

        # Check if we are in the first tensor parallel group.
        if self._text_output_stream is None and self._json_output_stream is None:
            return

        for id_, prompt, hypotheses in zip(ids, prompts, output.hypotheses):
            if len(hypotheses) == 0:
                raise InternalError(
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
            stream = self._text_output_stream
            if stream is not None:
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
            stream = self._json_output_stream
            if stream is not None:
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

        stream = self._text_output_stream
        if stream is not None:
            stream.flush()

        stream = self._json_output_stream
        if stream is not None:
            stream.flush()

    @property
    @override
    def metric_bag(self) -> SequenceGenerationMetricBag:
        return self._metric_bag
