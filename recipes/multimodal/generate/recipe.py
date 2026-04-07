# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import logging
from typing import Any, TextIO

from typing_extensions import override

from fairseq2.composition import register_dataset_family
from fairseq2.file_system import FileMode
from fairseq2.models.hg.api import load_hg_model_simple
from fairseq2.recipe.base import Recipe, RecipeContext
from fairseq2.runtime.dependency import DependencyContainer
from fairseq2.task import Task

from .config import MultimodalGenerateConfig
from .dataset import (
    MULTIMODAL_GENERATE_DATASET_FAMILY,
    MultimodalGenerateDataset,
    MultimodalGenerateDatasetConfig,
    open_multimodal_generate_dataset,
)
from .handlers import ModelHandler, get_handler

logger = logging.getLogger(__name__)


class MultimodalGenerateRecipe(Recipe):
    @override
    def register(self, container: DependencyContainer) -> None:
        register_dataset_family(
            container,
            MULTIMODAL_GENERATE_DATASET_FAMILY,
            MultimodalGenerateDataset,
            MultimodalGenerateDatasetConfig,
            opener=open_multimodal_generate_dataset,
        )

        # Register all handler model classes (safe — no heavy imports).
        from .handlers import _HANDLER_MAP

        for handler in _HANDLER_MAP.values():
            handler.register_model()

    @override
    def create_task(self, context: RecipeContext) -> Task:
        config = context.get_config_as(MultimodalGenerateConfig)

        gangs = context.gangs
        device = gangs.root.device

        handler = get_handler(config.model.handler, config.model.hf_name)

        logger.info(f"Loading model: {config.model.hf_name}")

        model = load_hg_model_simple(
            config.model.hf_name,
            use_processor=True,
            trust_remote_code=config.model.trust_remote_code,
            dtype=config.model.dtype,
        )

        model.to(device)
        handler.prepare_model(model)
        model.eval()

        processor = model.processor

        # Set up output files (only on tp rank 0).
        text_fp: TextIO | None = None
        json_fp: TextIO | None = None

        if gangs.tp.rank == 0:
            rank = gangs.dp.rank

            text_file = context.output_dir.joinpath(f"output/rank_{rank}.txt")
            json_file = context.output_dir.joinpath(f"output/rank_{rank}.jsonl")

            file_system = context.file_system
            file_system.make_directory(text_file.parent)

            text_fp = file_system.open_text(text_file, mode=FileMode.WRITE)
            json_fp = file_system.open_text(json_file, mode=FileMode.WRITE)

        # Create dataset reader.
        dataset = context.get_dataset_as(MultimodalGenerateDataset)

        data_reader = dataset.create_reader(
            gangs,
            batch_size=config.dataset.batch_size,
            prefetch=config.dataset.prefetch,
        )

        return MultimodalGenerateTask(
            handler=handler,
            model=model,
            processor=processor,
            data_reader=data_reader,
            device=device,
            num_frames=config.video.num_frames,
            max_new_tokens=config.generation.max_new_tokens,
            do_sample=config.generation.do_sample,
            temperature=config.generation.temperature,
            top_p=config.generation.top_p,
            text_fp=text_fp,
            json_fp=json_fp,
        )

    @property
    @override
    def config_kls(self) -> type[object]:
        return MultimodalGenerateConfig


class MultimodalGenerateTask(Task):
    def __init__(
        self,
        *,
        handler: ModelHandler,
        model: Any,
        processor: Any,
        data_reader: Any,
        device: Any,
        num_frames: int,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        text_fp: TextIO | None,
        json_fp: TextIO | None,
    ) -> None:
        self._handler = handler
        self._model = model
        self._processor = processor
        self._data_reader = data_reader
        self._device = device
        self._num_frames = num_frames
        self._max_new_tokens = max_new_tokens
        self._do_sample = do_sample
        self._temperature = temperature
        self._top_p = top_p
        self._text_fp = text_fp
        self._json_fp = json_fp
        self._step_nr = 0
        self._stop_requested = False

    @override
    def run(self) -> None:
        logger.info("Starting multimodal generation.")

        for batches in self._data_reader:
            if self._stop_requested:
                break

            for bucket in batches:
                if self._stop_requested:
                    break

                for example in bucket:
                    if self._stop_requested:
                        break

                    self._step_nr += 1

                    messages = example.get("messages", [])
                    example_id = example.get("id")

                    # Prepare inputs via handler.
                    inputs, prompt_text = self._handler.prepare_inputs(
                        self._processor,
                        messages,
                        self._num_frames,
                        self._device,
                    )

                    # Generate via handler.
                    output_ids = self._handler.generate(
                        self._model,
                        inputs,
                        max_new_tokens=self._max_new_tokens,
                        do_sample=self._do_sample,
                        temperature=self._temperature,
                        top_p=self._top_p,
                    )

                    # Decode via handler.
                    input_len = inputs["input_ids"].shape[-1]
                    response = self._handler.decode(
                        self._processor, output_ids, input_len
                    )

                    generated_len = output_ids.shape[-1] - input_len
                    logger.info(f"[step {self._step_nr}] Generated {generated_len} tokens.")

                    self._write_output(example_id, prompt_text, response)

        logger.info(f"Generation complete. {self._step_nr} examples processed.")

    def _write_output(
        self, example_id: Any, prompt: str, response: str
    ) -> None:
        # Write text output.
        stream = self._text_fp
        if stream is not None:
            if example_id is not None:
                stream.write("<<<<< ID >>>>>\n")
                stream.write(f"{example_id}\n\n")

            stream.write("<<<<< PROMPT >>>>>\n")
            stream.write(prompt)
            stream.write("\n\n<<<<< RESPONSE >>>>>\n")
            stream.write(response)
            stream.write("\n\n\n============================\n\n\n")
            stream.flush()

        # Write JSON output.
        stream = self._json_fp
        if stream is not None:
            json_output = {
                "id": example_id,
                "prompt": prompt,
                "response": response,
            }
            json.dump(json_output, stream, indent=None)
            stream.write("\n")
            stream.flush()

    @override
    def request_stop(self) -> None:
        self._stop_requested = True

    @property
    @override
    def step_nr(self) -> int:
        return self._step_nr

    def close(self) -> None:
        if self._text_fp is not None:
            self._text_fp.close()
        if self._json_fp is not None:
            self._json_fp.close()
