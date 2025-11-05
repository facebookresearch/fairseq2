# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OLMO2-specific RoPE implementation following HuggingFace."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq2.models.olmo2.config import OLMO2Config


def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input.

    This implementation follows HuggingFace's rotate_half function used in
    apply_rotary_pos_emb.

    Args:
        x: Input tensor

    Returns:
        Tensor with the same shape as input, with the two halves rotated.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors.

    This implementation follows HuggingFace's apply_rotary_pos_emb function.

    Args:
        q: The query tensor. Shape: (B, H, S, D) or (B, S, H, D)
        k: The key tensor. Shape: (B, H, S, D) or (B, S, H, D)
        cos: The cosine part of the rotary embedding. Shape: (B, S, D)
        sin: The sine part of the rotary embedding. Shape: (B, S, D)

    Returns:
        Tuple of (q_embed, k_embed) with rotary position embeddings applied.
    """
    q_type, k_type = q.dtype, k.dtype

    # Determine if we need to unsqueeze for (B, H, S, D) layout
    if q.ndim == 4 and q.shape[1] != cos.shape[1]:
        # Assume (B, H, S, D) layout - need to unsqueeze cos/sin at dim 1
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed.to(q_type), k_embed.to(k_type)


class OLMO2RotaryEmbedding(nn.Module):
    """OLMO2 Rotary Position Embedding following HuggingFace implementation to ensure compatibility.
    """

    inv_freq: Tensor  # fix linting for `register_buffer`

    def __init__(self, config: OLMO2Config, device: torch.device | None = None) -> None:
        """Initialize OLMO2 Rotary Embedding.

        Args:
            config: OLMO2 model configuration.
            device: Device to place the inverse frequency buffer on.
        """
        super().__init__()
        self.max_seq_len_cached = config.max_seq_len
        self.original_max_seq_len = config.max_seq_len

        self.config = config

        # Compute inverse frequencies for RoPE
        inv_freq, self.attention_scaling = self.compute_default_rope_parameters(
            self.config, device
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: OLMO2Config,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple[Tensor, float]:
        """Computes the inverse frequencies according to the original RoPE implementation.

        Args:
            config: The OLMO2 configuration.
            device: The device to use for initialization of the inverse frequencies.
            seq_len: The current sequence length. Unused for this type of RoPE.

        Returns:
            Tuple of (inv_freq, attention_factor), containing the inverse frequencies
            for the RoPE embeddings and the post-processing scaling factor applied to
            the computed cos/sin (unused in this type of RoPE, always 1.0).
        """
        base = config.rope_theta
        dim = config.model_dim // config.num_attn_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        # inv_freq = 1.0 / (base^(i/dim)) for i in [0, 2, 4, ..., dim-2]
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Compute cos and sin embeddings for the given positions.

        Args:
            x: Input tensor to determine dtype and device. Shape: (B, S, H, D) or (B, H, S, D)
            position_ids: Position indices. Shape: (B, S)

        Returns:
            Tuple of (cos, sin) tensors with shape (B, S, D) where D is head_dim.
        """
        # inv_freq: (D/2,)
        # position_ids: (B, S)

        # Expand inv_freq to (B, D/2, 1)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        ).to(x.device)

        # Expand position_ids to (B, 1, S)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 computation for numerical stability
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # Matrix multiply: (B, D/2, 1) @ (B, 1, S) -> (B, D/2, S)
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # freqs: (B, S, D/2)

            # Concatenate freqs with itself to get full dimension
            # (B, S, D/2) -> (B, S, D)
            emb = torch.cat((freqs, freqs), dim=-1)

            # Compute cos and sin
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos, sin
