# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Protocol


class ModelHandler(Protocol):
    """Protocol for model-family-specific multimodal generation logic."""

    def register_model(self) -> None:
        """Register the model class with fairseq2's HG model registry.

        Called during recipe registration. Should be safe to call even if the
        model weights are not present (no heavy imports).
        """
        ...

    def prepare_model(self, model: Any) -> None:
        """Apply model-specific fixups after loading (e.g. dtype casts)."""
        ...

    def prepare_inputs(
        self,
        processor: Any,
        messages: list[dict[str, Any]],
        num_frames: int,
        device: Any,
    ) -> tuple[dict[str, Any], str]:
        """Prepare model inputs from chat messages.

        :param processor: The HuggingFace processor for this model.
        :param messages: Chat messages with content blocks.
        :param num_frames: Number of frames to extract per video.
        :param device: Target device for tensors.
        :returns: (model_inputs, prompt_text) — inputs dict ready for
            ``model.generate()`` and the prompt string for logging.
        """
        ...

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
        """Run model.generate() with family-specific kwargs.

        :returns: Output token IDs tensor.
        """
        ...

    def decode(
        self,
        processor: Any,
        output_ids: Any,
        input_length: int,
    ) -> str:
        """Decode generated token IDs into text.

        :param processor: The HuggingFace processor.
        :param output_ids: Full output IDs tensor (including prompt).
        :param input_length: Length of the prompt tokens to skip.
        :returns: Decoded response string.
        """
        ...


_HANDLER_MAP: dict[str, ModelHandler] = {}


def _register_handler(name: str, handler: ModelHandler) -> None:
    _HANDLER_MAP[name] = handler


def get_handler(handler_name: str, hf_name: str) -> ModelHandler:
    """Get a model handler by name, or auto-detect from hf_name.

    :param handler_name: Explicit handler name, or ``"auto"`` to detect.
    :param hf_name: HuggingFace model identifier for auto-detection.
    :returns: A ``ModelHandler`` instance.
    :raises ValueError: If no handler matches.
    """
    if handler_name != "auto":
        if handler_name not in _HANDLER_MAP:
            raise ValueError(
                f"Unknown handler '{handler_name}'. "
                f"Available: {list(_HANDLER_MAP.keys())}"
            )
        return _HANDLER_MAP[handler_name]

    # Auto-detect from hf_name.
    hf_lower = hf_name.lower()

    if "gemma" in hf_lower:
        return _HANDLER_MAP["gemma3"]

    if "qwen" in hf_lower and "omni" in hf_lower:
        return _HANDLER_MAP["qwen_omni"]

    raise ValueError(
        f"Cannot auto-detect handler for model '{hf_name}'. "
        f"Set 'handler' explicitly in config. "
        f"Available: {list(_HANDLER_MAP.keys())}"
    )


# Import handler modules to trigger registration.
from . import gemma3 as _gemma3  # noqa: E402, F401
from . import qwen_omni as _qwen_omni  # noqa: E402, F401
