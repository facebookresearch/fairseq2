# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Relative position attention for the Parakeet audio encoder.

Implements Shaw-style relative position self-attention used in
NVIDIA's FastConformer/Parakeet architecture:

  score = (Q + bias_u) @ K^T + rel_shift((Q + bias_v) @ R^T)

where R is position-encoded via sinusoidal relative position embeddings
projected through relative_k_proj.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, final

import torch
from torch import Tensor
from torch.nn import Linear, Module, Parameter
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device


@final
class ParakeetRelativePositionalEncoding(Module):
    """Sinusoidal relative positional encoding for Parakeet.

    Holds an ``inv_freq`` buffer (non-persistent, like the HF model) and
    computes sinusoidal position embeddings on the fly during forward.
    """

    def __init__(
        self,
        model_dim: int,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        # inv_freq is a non-persistent buffer (not saved in state dict)
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, model_dim, 2, dtype=torch.float32) / model_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int) -> Tensor:
        """Compute sinusoidal position embeddings.

        :param seq_len:
            The sequence length.

        :returns:
            Position embeddings. *Shape:* ``[seq_len, model_dim]``.
        """
        # Create position indices [seq_len-1, seq_len-2, ..., 0]
        # (relative positions: most distant first)
        positions = torch.arange(
            seq_len - 1, -1, -1.0, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )

        # [seq_len, dim/2]
        sinusoid = torch.outer(positions, self.inv_freq)

        # [seq_len, dim] — interleave sin and cos
        pos_enc = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)

        return pos_enc

    if TYPE_CHECKING:
        __call__ = forward


def _rel_shift(x: Tensor) -> Tensor:
    """Perform relative shift (skew) operation for relative position attention.

    Converts the position-based attention scores into the correct alignment
    by padding and slicing.

    :param x:
        Tensor of shape ``[B, H, T, 2*T-1]`` or ``[B, H, T, T+pad]``.

    :returns:
        Tensor of shape ``[B, H, T, T]`` after relative shift.
    """
    b, h, t, _ = x.shape

    # Pad with one column on the left
    # [B, H, T, 2T-1] -> [B, H, T, 2T]
    x = torch.nn.functional.pad(x, (1, 0))

    # Reshape to allow diagonal extraction
    # [B, H, T, 2T] -> [B, H, 2T, T]
    x = x.view(b, h, -1, t)

    # Take first T rows (this performs the skew)
    # [B, H, 2T, T] -> [B, H, T, T]
    x = x[:, :, 1:, :]  # Skip first row (padding)
    x = x[:, :, :t, :]  # Take only T rows

    return x


@final
class ParakeetRelativeAttention(Module):
    """Shaw-style relative position multi-head self-attention for Parakeet.

    Uses per-head content (``bias_u``) and position (``bias_v``) biases:

        content_score = (Q + bias_u) @ K^T
        position_score = rel_shift((Q + bias_v) @ R^T)
        score = (content_score + position_score) / sqrt(head_dim)
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        *,
        head_dim: int | None = None,
        bias: bool = True,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (model_dim // num_heads)

        self.q_proj = Linear(model_dim, model_dim, bias=bias, device=device, dtype=dtype)
        self.k_proj = Linear(model_dim, model_dim, bias=bias, device=device, dtype=dtype)
        self.v_proj = Linear(model_dim, model_dim, bias=bias, device=device, dtype=dtype)
        self.output_proj = Linear(
            model_dim, model_dim, bias=bias, device=device, dtype=dtype
        )

        # Project position encoding to key space (no bias per HF checkpoint)
        self.relative_k_proj = Linear(
            model_dim, model_dim, bias=False, device=device, dtype=dtype
        )

        # Per-head biases for content and position
        self.bias_u = Parameter(
            torch.zeros(num_heads, self.head_dim, device=device, dtype=dtype)
        )
        self.bias_v = Parameter(
            torch.zeros(num_heads, self.head_dim, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor, pos_enc: Tensor) -> Tensor:
        """Forward pass with relative position attention.

        :param x:
            Input features. *Shape:* ``[B, T, D]``.
        :param pos_enc:
            Position encodings from ``ParakeetRelativePositionalEncoding``.
            *Shape:* ``[T, D]``.

        :returns:
            Attention output. *Shape:* ``[B, T, D]``.
        """
        b, t, _ = x.shape
        h = self.num_heads
        d = self.head_dim

        # Project to Q, K, V
        # [B, T, D] -> [B, T, H, d] -> [B, H, T, d]
        q = self.q_proj(x).view(b, t, h, d).transpose(1, 2)
        k = self.k_proj(x).view(b, t, h, d).transpose(1, 2)
        v = self.v_proj(x).view(b, t, h, d).transpose(1, 2)

        # Project position encoding to key space
        # [T, D] -> [T, D] -> [T, H, d] -> [H, T, d]
        rel_k = self.relative_k_proj(pos_enc).view(t, h, d).permute(1, 0, 2)

        # Content-based attention: (Q + bias_u) @ K^T
        # bias_u: [H, d] -> [1, H, 1, d]
        q_with_u = q + self.bias_u.unsqueeze(0).unsqueeze(2)
        content_score = torch.matmul(q_with_u, k.transpose(-2, -1))  # [B, H, T, T]

        # Position-based attention: rel_shift((Q + bias_v) @ R^T)
        q_with_v = q + self.bias_v.unsqueeze(0).unsqueeze(2)
        # [B, H, T, d] @ [H, d, T] -> [B, H, T, T]
        position_score = torch.matmul(q_with_v, rel_k.transpose(-2, -1))
        position_score = _rel_shift(position_score)

        # Combined score
        scale = 1.0 / math.sqrt(d)
        attn_weights = (content_score + position_score) * scale

        attn_weights = torch.softmax(attn_weights, dim=-1)

        # [B, H, T, T] @ [B, H, T, d] -> [B, H, T, d]
        attn_output = torch.matmul(attn_weights, v)

        # [B, H, T, d] -> [B, T, H, d] -> [B, T, D]
        attn_output = attn_output.transpose(1, 2).reshape(b, t, self.model_dim)

        return self.output_proj(attn_output)

    if TYPE_CHECKING:
        __call__ = forward

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"model_dim={self.model_dim}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}"
        )
