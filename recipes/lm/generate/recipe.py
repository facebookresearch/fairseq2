# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TextIO

from typing_extensions import override

from fairseq2.composition import register_dataset_family
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.datasets import SequenceBatch
from fairseq2.error import CorruptDataError, InternalError
from fairseq2.file_system import FileMode
from fairseq2.generation import SequenceGenerator
from fairseq2.metrics import MetricBag
from fairseq2.metrics.common import (
    add_seq_generator_metrics,
    update_seq_generator_metrics,
)
from fairseq2.recipe.base import Recipe, RecipeContext
from fairseq2.recipe.generator import GeneratorUnit
from fairseq2.recipe.task import Task
from fairseq2.runtime.dependency import DependencyContainer

from ..common import check_model_vocabulary
from .config import LMGenerateConfig
from .dataset import (
    LM_GENERATE_DATASET_FAMILY,
    LMGenerateDataset,
    LMGenerateDatasetConfig,
    open_lm_generate_dataset,
)


class LMGenerateRecipe(Recipe):
    @override
    def register(self, container: DependencyContainer) -> None:
        register_dataset_family(
            container,
            LM_GENERATE_DATASET_FAMILY,
            LMGenerateDataset,
            LMGenerateDatasetConfig,
            opener=open_lm_generate_dataset,
        )

    @override
    def create_task(self, context: RecipeContext) -> Task:
        check_model_vocabulary(context)

        config = context.get_config_as(LMGenerateConfig)

        gangs = context.gangs

        if gangs.tp.rank == 0:
            rank = gangs.dp.rank

            text_file = context.output_dir.joinpath(f"output/rank_{rank}.txt")
            json_file = context.output_dir.joinpath(f"output/rank_{rank}.jsonl")

            file_system = context.file_system

            file_system.make_directory(text_file.parent)

            text_fp = file_system.open_text(text_file, mode=FileMode.WRITE)
            json_fp = file_system.open_text(json_file, mode=FileMode.WRITE)
        else:
            text_fp = None
            json_fp = None

        seq_generator = context.get_seq_generator()

        tokenizer = context.get_tokenizer()

        unit = LMGenerateUnit(seq_generator, tokenizer, text_fp, json_fp)

        dataset = context.get_dataset_as(LMGenerateDataset)

        data_reader = dataset.create_reader(
            tokenizer,
            gangs,
            batch_size=config.dataset.batch_size,
            prefetch=config.dataset.prefetch,
        )

        return context.create_generation_task(unit, data_reader)

    @property
    @override
    def config_kls(self) -> type[object]:
        return LMGenerateConfig


class LMGenerateUnit(GeneratorUnit[SequenceBatch]):
    def __init__(
        self,
        generator: SequenceGenerator,
        tokenizer: Tokenizer,
        text_output_stream: TextIO | None,
        json_output_stream: TextIO | None,
    ) -> None:
        text_decoder = tokenizer.create_decoder(skip_special_tokens=False)

        self._generator = generator
        self._text_decoder = text_decoder
        self._text_output_stream = text_output_stream
        self._json_output_stream = json_output_stream

    @override
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        add_seq_generator_metrics(metric_bag)

    @override
    def process_batch(self, batch: SequenceBatch, metric_bag: MetricBag) -> None:
        if batch.example is None:
            raise CorruptDataError("`batch.example` is `None`.")

        if not isinstance(batch.example, Mapping):
            raise CorruptDataError(
                f"`batch.example` is expected to be of type `{Mapping}`, but is of type `{type(batch.example)}` instead."
            )

        prompts = batch.example.get("prompt")
        if prompts is None:
            raise CorruptDataError(
                "`batch.example` does not contain a key named 'prompt'."
            )

        if not isinstance(prompts, Sequence):
            raise CorruptDataError(
                f"`batch.example['prompt']` is expected to be of type `{Sequence}`, but is of type `{type(prompts)}` instead."
            )

        ids = batch.example.get("id")
        if ids is None:
            raise CorruptDataError("`batch.example` does not contain a key named 'id'.")

        prompt_seqs, prompt_seqs_layout = batch.as_input()

        try:
            output = self._generator(prompt_seqs, prompt_seqs_layout)
        except SequenceGenerationError as ex:
            raise ModelError("Model produced NaN during sequence generation.") from ex

        update_seq_generator_metrics(metric_bag, output)

        # Check if we are in the first tensor parallel group.
        if self._text_output_stream is None and self._json_output_stream is None:
            return

        for id_, prompt, hypotheses in zip(ids, prompts, output.hypotheses):
            if len(hypotheses) == 0:
                raise InternalError("Sequence generator returned no hypothesis.")

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
