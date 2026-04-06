# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import logging
from typing import Any, TextIO

import torch
from typing_extensions import override

from fairseq2.composition import register_dataset_family
from fairseq2.file_system import FileMode
from fairseq2.models.hg.api import load_hg_model_simple
from fairseq2.models.hg.factory import register_hg_model_class
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
from .video import prepare_multimodal_messages

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

        # Register Gemma3 as a special model so _load_special_model is used
        # (same pattern as Qwen2.5 Omni in factory.py).
        register_hg_model_class(
            "Gemma3Config",
            "Gemma3ForConditionalGeneration",
            processor_class="Gemma3Processor",
        )

    @override
    def create_task(self, context: RecipeContext) -> Task:
        config = context.get_config_as(MultimodalGenerateConfig)

        gangs = context.gangs
        device = gangs.root.device

        logger.info(f"Loading model: {config.model.hf_name}")

        model = load_hg_model_simple(
            config.model.hf_name,
            use_processor=True,
            trust_remote_code=config.model.trust_remote_code,
            dtype=config.model.dtype,
        )

        model.to(device)
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

                    # Expand video/image blocks into image blocks + collect PIL images.
                    rewritten_msgs, images = prepare_multimodal_messages(
                        messages, self._num_frames
                    )

                    # Apply chat template to get prompt text.
                    prompt_text = self._processor.apply_chat_template(
                        rewritten_msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                    # Tokenize + create pixel_values via processor.
                    if images:
                        inputs = self._processor(
                            text=prompt_text,
                            images=images,
                            return_tensors="pt",
                        )
                    else:
                        inputs = self._processor(
                            text=prompt_text,
                            return_tensors="pt",
                        )

                    inputs = {k: v.to(self._device) for k, v in inputs.items()}

                    # Generate.
                    with torch.inference_mode():
                        output_ids = self._model.generate(
                            **inputs,
                            max_new_tokens=self._max_new_tokens,
                            do_sample=self._do_sample,
                            temperature=self._temperature if self._do_sample else None,
                            top_p=self._top_p if self._do_sample else None,
                        )

                    # Decode only the generated tokens (skip prompt tokens).
                    input_len = inputs["input_ids"].shape[-1]
                    generated_ids = output_ids[0, input_len:]
                    response = self._processor.decode(
                        generated_ids, skip_special_tokens=True
                    )

                    logger.info(f"[step {self._step_nr}] Generated {len(generated_ids)} tokens.")

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
