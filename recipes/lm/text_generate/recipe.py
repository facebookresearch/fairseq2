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

from fairseq2.composition import register_dataset_family
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.datasets import SequenceBatch
from fairseq2.error import InternalError, raise_operational_system_error
from fairseq2.file_system import FileMode
from fairseq2.generation import SequenceGenerator
from fairseq2.metrics import MetricBag
from fairseq2.metrics.common import (
    add_seq_generator_metrics,
    update_seq_generator_metrics,
)
from fairseq2.recipe.base import GenerationRecipe, RecipeContext
from fairseq2.recipe.config import (
    CommonSection,
    DatasetSection,
    GangSection,
    GeneratorSection,
    ReferenceModelSection,
    SequenceGeneratorSection,
    TokenizerSection,
)
from fairseq2.recipe.generator import Generator, GeneratorUnit
from fairseq2.recipe.model import RecipeModel
from fairseq2.runtime.dependency import DependencyContainer

from ..common import check_model_vocabulary
from .dataset import (
    TEXT_GEN_DATASET_FAMILY,
    TextGenDataset,
    TextGenDatasetConfig,
    open_text_gen_dataset,
)


@dataclass(kw_only=True)
class TextGenConfig:
    model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(
            name="llama3_2_1b_instruct", dtype=torch.bfloat16
        )
    )

    dataset: TextGenDatasetSection = field(
        default_factory=lambda: TextGenDatasetSection(
            family=TEXT_GEN_DATASET_FAMILY,
            config_overrides=TextGenDatasetConfig(
                paths=[Path("~/train.jsonl")],
            ),
        )
    )

    tokenizer: TokenizerSection = field(
        default_factory=lambda: TokenizerSection(name="llama3_instruct")
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    generator: GeneratorSection = field(default_factory=lambda: GeneratorSection())

    seq_generator: SequenceGeneratorSection = field(
        default_factory=lambda: SequenceGeneratorSection()
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())


@dataclass(kw_only=True)
class TextGenDatasetSection(DatasetSection):
    batch_size: int = 1
    prefetch: int = 4


@final
class TextGenRecipe(GenerationRecipe):
    @override
    def register(self, container: DependencyContainer) -> None:
        register_dataset_family(
            container,
            TEXT_GEN_DATASET_FAMILY,
            TextGenDataset,
            TextGenDatasetConfig,
            opener=open_text_gen_dataset,
        )

    @override
    def create_generator(self, context: RecipeContext) -> Generator:
        check_model_vocabulary(context)

        config = context.config.as_(TextGenConfig)

        gangs = context.gangs

        if gangs.tp.rank == 0:
            rank = gangs.dp.rank

            text_file = context.output_dir.joinpath(f"output/rank_{rank}.txt")
            json_file = context.output_dir.joinpath(f"output/rank_{rank}.jsonl")

            file_system = context.file_system

            try:
                file_system.make_directory(text_file.parent)
            except OSError as ex:
                raise_operational_system_error(ex)

            try:
                text_fp = file_system.open_text(text_file, mode=FileMode.WRITE)
                json_fp = file_system.open_text(json_file, mode=FileMode.WRITE)
            except OSError as ex:
                raise_operational_system_error(ex)
        else:
            text_fp = None
            json_fp = None

        unit = TextGenUnit(
            context.model,
            context.default_seq_generator,
            context.default_tokenizer,
            text_output_stream=text_fp,
            json_output_stream=json_fp,
        )

        dataset = context.default_dataset.as_(TextGenDataset)

        data_reader = dataset.create_reader(
            context.default_tokenizer,
            gangs,
            batch_size=config.dataset.batch_size,
            prefetch=config.dataset.prefetch,
        )

        return context.create_generator(unit, data_reader)

    @property
    @override
    def config_kls(self) -> type[object]:
        return TextGenConfig


@final
class TextGenUnit(GeneratorUnit[SequenceBatch]):
    def __init__(
        self,
        model: RecipeModel,
        generator: SequenceGenerator,
        tokenizer: Tokenizer,
        text_output_stream: TextIO | None,
        json_output_stream: TextIO | None,
    ) -> None:
        self._model = model
        self._generator = generator
        self._text_decoder = tokenizer.create_decoder(skip_special_tokens=False)
        self._text_output_stream = text_output_stream
        self._json_output_stream = json_output_stream

    @override
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        add_seq_generator_metrics(metric_bag)

    @override
    def process_batch(self, batch: SequenceBatch, metric_bag: MetricBag) -> None:
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
                f"`batch.example['prompt']` must be a sequence of strings, but is of type `{type(prompts)}` instead."
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
                        "The sequence generator returned no hypothesis."
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
            raise_operational_system_error(ex)

    @property
    @override
    def model(self) -> RecipeModel:
        return self._model
