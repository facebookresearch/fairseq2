# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO, final

import torch
from typing_extensions import override

from fairseq2.data.tokenizers import TokenDecoder, Tokenizer
from fairseq2.datasets import (
    SequenceBatch,
    StaticBatching,
    SyncMode,
    register_dataset_family,
)
from fairseq2.error import InfraError, InternalError
from fairseq2.file_system import FileMode
from fairseq2.generation import SequenceGenerator
from fairseq2.generator import Generator, GeneratorUnit
from fairseq2.metrics import MetricBag
from fairseq2.metrics.common import update_seq_generator_metrics
from fairseq2.model.context import ModelContext
from fairseq2.models.clm import CausalLM
from fairseq2.recipe.base import GenerationRecipe, RecipeContext
from fairseq2.recipe.config import (
    CommonSection,
    DatasetSectionBase,
    GangSection,
    GeneratorSection,
    ReferenceModelSection,
    SequenceGeneratorSection,
    TokenizerSection,
)
from fairseq2.runtime.dependency import DependencyContainer

from .dataset import (
    INSTRUCTION_DATASET_FAMILY,
    InstructionDataset,
    InstructionPromptReadOptions,
)


@dataclass(kw_only=True)
class TextGenerationConfig:
    model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(name="llama3_2_1b_instruct")
    )

    dataset: TextGenerateDatasetSection = field(
        default_factory=lambda: TextGenerateDatasetSection()
    )

    tokenizer: TokenizerSection = field(
        default_factory=lambda: TokenizerSection(name="llama3_instruct")
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    generator: GeneratorSection = field(
        default_factory=lambda: GeneratorSection(dtype=torch.bfloat16)
    )

    seq_generator: SequenceGeneratorSection = field(
        default_factory=lambda: SequenceGeneratorSection()
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())


@dataclass(kw_only=True)
class TextGenerateDatasetSection(DatasetSectionBase):
    name: str = "foo"  # TODO: change!

    family: str = INSTRUCTION_DATASET_FAMILY

    path: Path | None = None

    split: str = "default"

    min_seq_len: int = 1
    """The minimum sequence length."""

    max_seq_len: int = 8192
    """The maximum sequence length."""

    batch_size: int = 1

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    extras: dict[str, object] = field(default_factory=dict)
    """The dataset-specific extra options."""


@final
class TextGenerationRecipe(GenerationRecipe):
    @override
    def register(self, container: DependencyContainer) -> None:
        register_dataset_family(
            container,
            INSTRUCTION_DATASET_FAMILY,
            InstructionDataset,
            InstructionDataset.from_path,
        )

    @override
    def create_generator(self, context: RecipeContext) -> Generator:
        gangs = context.gangs

        if gangs.tp.rank == 0:
            output_dir = context.output_dir

            file_system = context.file_system

            rank = gangs.dp.rank

            text_file = output_dir.joinpath(f"output/rank_{rank}.txt")
            json_file = output_dir.joinpath(f"output/rank_{rank}.jsonl")

            try:
                file_system.make_directory(text_file.parent)
            except OSError as ex:
                raise InfraError(
                    f"The '{text_file.parent}' output directory cannot be created. See the nested exception for details."
                ) from ex

            try:
                text_fp = file_system.open_text(text_file, mode=FileMode.WRITE)
            except OSError as ex:
                raise InfraError(
                    f"The '{text_file}' output file cannot be created. See the nested exception for details."
                ) from ex

            try:
                json_fp = file_system.open_text(json_file, mode=FileMode.WRITE)
            except OSError as ex:
                raise InfraError(
                    f"The '{json_file}' output file cannot be created. See the nested exception for details."
                ) from ex
        else:
            text_fp = None
            json_fp = None

        tokenizer = context.tokenizer

        unit = TextGeneratorUnit(
            context.model_context,
            context.seq_generator,
            tokenizer,
            text_output_stream=text_fp,
            json_output_stream=json_fp,
        )

        dataset = context.dataset_as(InstructionDataset)

        config = context.config_as(TextGenerationConfig)

        seed = context.next_seed()

        batching = StaticBatching(config.dataset.batch_size)

        read_options = InstructionPromptReadOptions(
            batching=batching,
            sync_mode=SyncMode.UNTIL_LAST,
            num_prefetch=config.dataset.num_prefetch,
            seed=seed,
            extras=config.dataset.extras,
        )

        data_reader = dataset.create_prompt_reader(
            config.dataset.split,
            tokenizer,
            gangs.dp,
            config.dataset.min_seq_len,
            config.dataset.max_seq_len,
            read_options,
        )

        return context.create_generator(unit, data_reader)

    @property
    @override
    def config_kls(self) -> type[object]:
        return TextGenerationConfig


@final
class TextGeneratorUnit(GeneratorUnit[SequenceBatch]):
    """Represents a text generation unit."""

    _model_context: ModelContext
    _generator: SequenceGenerator
    _text_decoder: TokenDecoder
    _text_output_stream: TextIO | None
    _json_output_stream: TextIO | None

    def __init__(
        self,
        model_context: ModelContext,
        generator: SequenceGenerator,
        tokenizer: Tokenizer,
        text_output_stream: TextIO | None,
        json_output_stream: TextIO | None,
    ) -> None:
        self._model_context = model_context

        self._generator = generator

        self._text_decoder = tokenizer.create_decoder(skip_special_tokens=False)

        self._text_output_stream = text_output_stream
        self._json_output_stream = json_output_stream

    @override
    def __call__(self, batch: SequenceBatch, metric_bag: MetricBag) -> None:
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

        if not isinstance(prompts, Sequence):
            raise TypeError(
                f"`batch.example['prompt'] must be a sequence of strings, but is of type `{type(prompts)}` instead."
            )

        ids = batch.example["id"]

        prompt_seqs, prompt_seqs_layout = batch.as_input()

        output = self._generator(prompt_seqs, prompt_seqs_layout)

        update_seq_generator_metrics(metric_bag, output)

        # Check if we are in the first tensor parallel group.
        if self._text_output_stream is None and self._json_output_stream is None:
            return

        try:
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
        except OSError as ex:
            raise InfraError(
                "The generator output cannot be written. See the nested exception for details."
            ) from ex

    @property
    @override
    def model_context(self) -> ModelContext:
        return self._model_context
