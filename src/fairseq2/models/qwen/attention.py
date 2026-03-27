# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gated multi-head attention for Qwen 3.5.

Differs from ``StandardMultiheadAttention`` in three ways:

1. The Q projection is doubled — half is the query, half is an output gate.
2. Partial RoPE: only the first ``encoding_dim`` dimensions are rotated.
3. Output gating: ``attn_output = attn_output * sigmoid(gate)``.

Reference: HuggingFace ``modeling_qwen3_5.py`` ``Qwen3_5Attention`` lines 707-779.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import torch
from torch import Tensor, nn

from fairseq2.models.transformer import (
    AttentionBiasCache,
    AttentionState,
    AttentionStateFactory,
    FullAttentionState,
    MultiheadAttention,
    SDPA,
)
from fairseq2.nn import (
    BatchLayout,
    IncrementalStateBag,
    LayerNorm,
    Linear,
    PositionEncoder,
)
from fairseq2.ops import repeat_interleave


class Qwen35Attention(MultiheadAttention):
    """Gated multi-head attention for Qwen 3.5 full-attention layers.

    Key differences from :class:`StandardMultiheadAttention`:

    * **Doubled Q projection** — ``q_proj`` outputs ``num_heads * head_dim * 2``;
      the second half is an output gate.
    * **Partial RoPE** — only the first ``encoding_dim`` (typically 64) of the
      ``head_dim`` (typically 256) are rotated. The rest pass through.
    * **Output gating** — ``attn_output * sigmoid(gate)`` before ``output_proj``.
    * **QK-Norm** on per-head dimension (after unflatten).

    Reference: ``modeling_qwen3_5.py`` lines 707-779.
    """

    num_heads: Final[int]
    num_key_value_heads: Final[int]
    num_query_groups: Final[int]
    head_dim: Final[int]

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        sdpa: SDPA,
        *,
        head_dim: int = 256,
        num_key_value_heads: int | None = None,
        pos_encoder: PositionEncoder | None = None,
        q_norm: LayerNorm | None = None,
        k_norm: LayerNorm | None = None,
        state_factory: AttentionStateFactory | None = None,
        qkv_proj_init_fn: Callable[[Linear], None] | None = None,
        output_proj_init_fn: Callable[[Linear], None] | None = None,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim

        if num_key_value_heads is None:
            num_key_value_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_query_groups = num_heads // num_key_value_heads

        # Q projection is DOUBLED — half query, half gate.
        # HF: nn.Linear(hidden, num_heads * head_dim * 2, bias=False)
        self.q_proj = Linear(
            model_dim, num_heads * head_dim * 2, bias=False,
            init_fn=qkv_proj_init_fn,
        )

        self.k_proj = Linear(
            model_dim, num_key_value_heads * head_dim, bias=False,
            init_fn=qkv_proj_init_fn,
        )

        self.v_proj = Linear(
            model_dim, num_key_value_heads * head_dim, bias=False,
            init_fn=qkv_proj_init_fn,
        )

        self.output_proj = Linear(
            num_heads * head_dim, model_dim, bias=False,
            init_fn=output_proj_init_fn,
        )

        self.q_norm = q_norm
        self.k_norm = k_norm
        self.pos_encoder = pos_encoder
        self.sdpa = sdpa
        self.state_factory = state_factory

    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        keys: Tensor,
        keys_layout: BatchLayout,
        values: Tensor,
        bias_cache: AttentionBiasCache,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        # -- Q projection: split into query + gate --
        # (B, S, num_heads * head_dim * 2)
        q_combined = self.q_proj(seqs)

        # (B, S, num_heads, head_dim * 2) -> split along last dim
        q_combined = q_combined.unflatten(-1, (self.num_heads, self.head_dim * 2))
        q, gate = q_combined.chunk(2, dim=-1)
        # q: (B, S, num_heads, head_dim)
        # gate: (B, S, num_heads, head_dim)

        # Flatten gate to (B, S, num_heads * head_dim) for later element-wise gating.
        gate = gate.flatten(-2)  # (B, S, num_heads * head_dim)

        # -- K, V projections --
        k = self.k_proj(keys)
        v = self.v_proj(values)
        k = k.unflatten(-1, (self.num_key_value_heads, self.head_dim))
        v = v.unflatten(-1, (self.num_key_value_heads, self.head_dim))

        # -- QK-Norm (per head dim, after unflatten) --
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        # -- Partial RoPE --
        # Only the first `encoding_dim` dimensions of each head are rotated.
        # The rest pass through unchanged.
        if self.pos_encoder is not None:
            encoding_dim = self.pos_encoder.encoding_dim

            if encoding_dim < self.head_dim:
                # Split into rotary and pass-through parts.
                q_rot = q[..., :encoding_dim]
                q_pass = q[..., encoding_dim:]
                k_rot = k[..., :encoding_dim]
                k_pass = k[..., encoding_dim:]

                q_rot = self.pos_encoder(
                    q_rot, seqs_layout, state_bag=state_bag
                )
                k_rot = self.pos_encoder(
                    k_rot, keys_layout, state_bag=state_bag
                )

                q = torch.cat([q_rot, q_pass], dim=-1)
                k = torch.cat([k_rot, k_pass], dim=-1)
            else:
                # Full rotation (encoding_dim == head_dim).
                q = self.pos_encoder(q, seqs_layout, state_bag=state_bag)
                k = self.pos_encoder(k, keys_layout, state_bag=state_bag)

        # -- KV cache management --
        if not self.training and state_bag is not None:
            state = state_bag.maybe_get_state(self, AttentionState)
            if state is None:
                state_factory = self.state_factory or FullAttentionState
                state = state_factory(
                    k, v, state_bag.max_num_steps, state_bag.capacity_increment
                )
                state_bag.set_state(self, state)
            else:
                state.append(k, v)
            k, v = state.get()
            keys_layout = BatchLayout.of(k)

        # -- GQA expansion --
        if self.num_query_groups > 1:
            k = repeat_interleave(k, dim=-2, repeat=self.num_query_groups)
            v = repeat_interleave(v, dim=-2, repeat=self.num_query_groups)

        # -- Scaled dot-product attention --
        # q, k, v: (B, S, H, D)
        attn_output, _ = self.sdpa(
            q, seqs_layout, k, keys_layout, v, bias_cache
        )

        # -- Output gating --
        # attn_output: (B, S, H, D) -> (B, S, H * D)
        attn_output = attn_output.flatten(-2)
        attn_output = attn_output * torch.sigmoid(gate)

        # -- Output projection --
        return self.output_proj(attn_output)
