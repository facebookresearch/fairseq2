# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from ..video import extract_frames
from . import _register_handler


class QwenOmniHandler:
    """Handler for Qwen 2.5 Omni models.

    Videos are pre-decoded with decord and passed as numpy arrays to avoid
    a torchcodec/FFmpeg dependency.
    """

    def register_model(self) -> None:
        # Qwen2.5 Omni is already in _HF_SPECIAL_MODELS in factory.py.
        pass

    def prepare_model(self, model: Any) -> None:
        # Flash Attention 2 requires an explicit dtype; cast the whole model to
        # bfloat16 so the warning about "no torch dtype" goes away.  The
        # Token2Wav sub-model will internally fall back to SDPA (expected).
        model.to(torch.bfloat16)

    def prepare_inputs(
        self,
        processor: Any,
        messages: list[dict[str, Any]],
        num_frames: int,
        device: Any,
    ) -> tuple[dict[str, Any], str]:
        # Apply chat template — Qwen's template handles video/audio natively.
        prompt_text: str = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Collect media from message blocks.
        # Videos are pre-decoded into numpy arrays via decord to avoid
        # the torchcodec/FFmpeg dependency that Qwen's processor uses.
        images: list[Any] = []
        videos: list[np.ndarray] = []
        audios: list[str] = []

        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                block_type = block.get("type")
                if block_type == "image":
                    url = block.get("url")
                    if url:
                        images.append(Image.open(url).convert("RGB"))
                elif block_type == "video":
                    url = block.get("url")
                    if url:
                        frames = extract_frames(url, num_frames)
                        # Stack PIL frames into (N, H, W, C) numpy array.
                        videos.append(np.stack([np.array(f) for f in frames]))
                elif block_type == "audio":
                    url = block.get("url")
                    if url:
                        audios.append(url)

        # Build processor kwargs — only pass non-empty media lists.
        proc_kwargs: dict[str, Any] = {
            "text": prompt_text,
            "return_tensors": "pt",
            "padding": True,
        }
        if images:
            proc_kwargs["images"] = images
        if videos:
            proc_kwargs["videos"] = videos
        if audios:
            proc_kwargs["audios"] = audios

        inputs = processor(**proc_kwargs)
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
                return_audio=False,
                use_audio_in_video=True,
                thinker_max_new_tokens=max_new_tokens,
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
        return processor.batch_decode(
            generated_ids.unsqueeze(0),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


_register_handler("qwen_omni", QwenOmniHandler())
