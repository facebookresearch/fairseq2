# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(kw_only=True)
class Gemma4AudioConfig:
    """Configuration for the Gemma 4 audio pipeline.

    Default values correspond to the E4B model (Conformer-based mel pipeline).

    The ``audio_mode`` field selects between the two Gemma 4 audio pipelines:

    * ``"conformer"`` (default; used by E4B / classic Gemma 4): mel-spectrogram
      input goes through a subsampling Conv2d stack and 12 Conformer layers
      (the audio tower) before the multimodal embedder projects to text space.

    * ``"linear"`` (used by Gemma 4 Unified family — 12B+): raw waveform frames
      of ``audio_samples_per_token`` (640) samples each are fed directly through
      the multimodal embedder (RMSNorm + Linear) — no tower, no mel, no convs.
      The ``hidden_size``, ``num_hidden_layers``, conv/attention etc. fields
      are ignored in this mode; only ``output_proj_dims`` (= 640 for unified)
      and ``rms_norm_eps`` are read.
    """

    audio_mode: Literal["conformer", "linear"] = "conformer"
    """Audio pipeline selector: ``"conformer"`` or ``"linear"``."""

    hidden_size: int = 1024
    """Audio encoder hidden dimension. (conformer mode only)"""

    output_proj_dims: int = 1536
    """Output projection dimension (before text embedder).

    For the unified ``linear`` mode this equals the raw-waveform frame size
    (typically 640 samples = 40 ms @ 16 kHz)."""

    num_hidden_layers: int = 12
    """Number of conformer layers. (conformer mode only)"""

    num_attention_heads: int = 8
    """Number of attention heads. head_dim = hidden_size / num_attention_heads."""

    conv_kernel_size: int = 5
    """Depthwise convolution kernel size in conformer."""

    residual_weight: float = 0.5
    """Macaron-style FFN residual scaling factor."""

    attention_chunk_size: int = 12
    """Chunk size for chunked local attention."""

    attention_context_left: int = 13
    """Left context (including current chunk) for local attention."""

    attention_context_right: int = 0
    """Right context for local attention (0 = causal)."""

    attention_logit_cap: float = 50.0
    """Pre-softmax logit softcapping value."""

    rms_norm_eps: float = 1e-6
    """Epsilon for RMSNorm layers."""

    gradient_clipping: float = 1e10
    """Gradient clipping value for conformer blocks."""

    subsampling_conv_channels: tuple[int, int] = (128, 32)
    """Output channels for the two subsample Conv2d layers."""

    input_feat_size: int = 128
    """Input feature size (mel-spectrogram channels)."""
