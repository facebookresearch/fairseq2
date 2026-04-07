# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import torch

from fairseq2.models.hg.factory import register_hg_model_class

from ..video import prepare_multimodal_messages
from . import _register_handler


class Gemma3Handler:
    """Handler for Gemma 3 vision-language models."""

    def register_model(self) -> None:
        register_hg_model_class(
            "Gemma3Config",
            "Gemma3ForConditionalGeneration",
            processor_class="Gemma3Processor",
        )

    def prepare_model(self, model: Any) -> None:
        pass

    def prepare_inputs(
        self,
        processor: Any,
        messages: list[dict[str, Any]],
        num_frames: int,
        device: Any,
    ) -> tuple[dict[str, Any], str]:
        # Expand video/image blocks into image blocks + collect PIL images.
        rewritten_msgs, images = prepare_multimodal_messages(messages, num_frames)

        # Apply chat template to get prompt text.
        prompt_text: str = processor.apply_chat_template(
            rewritten_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize + create pixel_values via processor.
        if images:
            inputs = processor(
                text=prompt_text,
                images=images,
                return_tensors="pt",
            )
        else:
            inputs = processor(
                text=prompt_text,
                return_tensors="pt",
            )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs, prompt_text

    def generate(
        self,
        model: Any,
        inputs: dict[str, Any],
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> Any:
        with torch.inference_mode():
            return model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
            )

    def decode(
        self,
        processor: Any,
        output_ids: Any,
        input_length: int,
    ) -> str:
        generated_ids = output_ids[0, input_length:]
        return processor.decode(generated_ids, skip_special_tokens=True)


_register_handler("gemma3", Gemma3Handler())
