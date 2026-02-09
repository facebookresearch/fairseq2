# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Any

from fairseq2.models.gemma3n.config import Gemma3nConfig


def load_gemma3n_checkpoint(
    checkpoint_path: str | Path,
    config: Gemma3nConfig,
) -> dict[str, Any]:
    """Load a Gemma3n checkpoint from HuggingFace format.

    Args:
        checkpoint_path: Path to the checkpoint directory or file.
        config: The Gemma3n configuration.

    Returns:
        The loaded state dictionary in fairseq2 format.

    Notes:
        This is a stub implementation for Phase 1. Full checkpoint loading
        with automatic config detection and weight conversion will be
        implemented in Phase 2-3.
    """
    # TODO(Phase 2): Implement HuggingFace checkpoint loading
    # - Detect checkpoint format (safetensors, pytorch_model.bin)
    # - Load config.json and validate against Gemma3nConfig
    # - Load weights with proper device placement
    # TODO(Phase 3): Add support for sharded checkpoints
    # TODO(Phase 3): Add support for quantized checkpoints

    raise NotImplementedError(
        "Checkpoint loading not yet implemented. "
        "Use direct state_dict conversion in Phase 3."
    )
