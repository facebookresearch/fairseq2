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

Position encoding generates 2*T-1 positions (both positive and negative
relative distances) with interleaved sin/cos:
  [sin(f0*p), cos(f0*p), sin(f1*p), cos(f1*p), ...]
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, final

import torch
from torch import Tensor
from torch.nn import Linear, Module, Parameter
from torch.nn.functional import pad
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device


@final
class ParakeetRelativePositionalEncoding(Module):
    """Sinusoidal relative positional encoding for Parakeet.

    Generates 2*seq_len - 1 positions (from seq_len-1 to -(seq_len-1))
    with interleaved sin/cos encoding, matching HF ParakeetEncoder.

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
            Position embeddings with interleaved sin/cos.
            *Shape:* ``[2*seq_len - 1, model_dim]``.
        """
        # Create position indices [seq_len-1, seq_len-2, ..., 0, -1, ..., -(seq_len-1)]
        # This covers both positive and negative relative distances.
        positions = torch.arange(
            seq_len - 1, -seq_len, -1.0,
            device=self.inv_freq.device, dtype=torch.float32,
        )

        # [2*seq_len-1, dim/2]
        freqs = torch.outer(positions, self.inv_freq)

        # Interleave sin and cos: [sin(f0), cos(f0), sin(f1), cos(f1), ...]
        # This matches HF's torch.stack([sin, cos], dim=-1).reshape(...)
        sin = freqs.sin()
        cos = freqs.cos()
        pos_enc = torch.stack([sin, cos], dim=-1).reshape(
            2 * seq_len - 1, -1
        )

        return pos_enc

    if TYPE_CHECKING:
        __call__ = forward


def _rel_shift(x: Tensor) -> Tensor:
    """Perform relative shift (skew) operation for relative position attention.

    Converts the position-based attention scores into the correct alignment
    by padding, reshaping, and slicing. Works for any position length P
    (typically P = 2*T - 1 for full relative positions).

    :param x:
        Tensor of shape ``[B, H, T, P]`` where P is the position length.

    :returns:
        Tensor of shape ``[B, H, T, P]`` after relative shift.
    """
    b, h, t, p = x.shape

    # Pad with one column on the left
    # [B, H, T, P] -> [B, H, T, P+1]
    x = pad(x, (1, 0))

    # Reshape to allow diagonal extraction
    # [B, H, T, P+1] -> [B, H, P+1, T]
    x = x.view(b, h, p + 1, t)

    # Skip first row (padding) to perform the skew
    # [B, H, P+1, T] -> [B, H, P, T]
    x = x[:, :, 1:]

    # Reshape back
    # [B, H, P, T] -> [B, H, T, P]
    x = x.view(b, h, t, p)

    return x


@final
class ParakeetRelativeAttention(Module):
    """Shaw-style relative position multi-head self-attention for Parakeet.

    Uses per-head content (``bias_u``) and position (``bias_v``) biases:

        content_score = (Q + bias_u) @ K^T * scale
        position_score = rel_shift((Q + bias_v) @ R^T)[:, :, :, :T] * scale
        score = content_score + position_score

    Position embeddings have shape [2*T-1, D] to cover both positive and
    negative relative distances, matching HF ParakeetEncoder.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        *,
        head_dim: int | None = None,
        bias: bool = False,
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
            *Shape:* ``[2*T-1, D]``.

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
        # pos_enc: [2T-1, D] -> [2T-1, D] -> [2T-1, H, d]
        # Cast to model dtype (pos_enc is computed in float32 for accuracy)
        p = pos_enc.shape[0]  # 2*T - 1
        rel_k = self.relative_k_proj(pos_enc.to(dtype=q.dtype)).view(p, h, d)

        # Content-based attention: (Q + bias_u) @ K^T * scale
        # bias_u: [H, d] -> [1, H, 1, d]
        q_with_u = q + self.bias_u.unsqueeze(0).unsqueeze(2)
        scale = 1.0 / math.sqrt(d)
        content_score = torch.matmul(q_with_u, k.transpose(-2, -1))  # [B, H, T, T]
        content_score = content_score * scale

        # Position-based attention: rel_shift((Q + bias_v) @ R^T)[:T] * scale
        q_with_v = q + self.bias_v.unsqueeze(0).unsqueeze(2)
        # [B, H, T, d] @ [1, H, d, 2T-1] -> [B, H, T, 2T-1]
        rel_k_transposed = rel_k.permute(1, 2, 0).unsqueeze(0)  # [1, H, d, 2T-1]
        position_score = torch.matmul(q_with_v, rel_k_transposed)
        position_score = _rel_shift(position_score)
        position_score = position_score[..., :t]  # [B, H, T, T]
        position_score = position_score * scale

        # Combined score
        attn_weights = content_score + position_score

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
