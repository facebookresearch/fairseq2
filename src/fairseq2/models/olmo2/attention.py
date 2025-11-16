# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OLMO2-specific attention module with Q/K normalization."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from fairseq2.gang import Gangs
from fairseq2.models.olmo2.rope import OLMO2RotaryEmbedding, apply_rotary_pos_emb
from fairseq2.models.transformer import StandardMultiheadAttention
from fairseq2.models.transformer.attention_bias import AttentionBiasCache
from fairseq2.models.transformer.sdpa.base import SDPA
from fairseq2.nn import (
    BatchLayout,
    IncrementalStateBag,
    LayerNorm,
    Linear,
    PositionEncoder,
    Projection,
)
from fairseq2.data_type import DataType
from fairseq2.device import Device


class OLMO2MultiheadAttention(StandardMultiheadAttention):
    """OLMO2 Multi-head Attention with Q/K normalization and HuggingFace-style RoPE.

    This class extends StandardMultiheadAttention to:
    1. Apply Q/K normalization on the full projected dimensions (before splitting into heads)
    2. Apply RoPE in the HuggingFace style (using rotate_half) instead of complex numbers
    """

    rope_module: OLMO2RotaryEmbedding | None

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        sdpa: SDPA,
        *,
        head_dim: int | None = None,
        num_key_value_heads: int | None = None,
        kv_dim: int | None = None,
        q_proj: Projection | None = None,
        k_proj: Projection | None = None,
        v_proj: Projection | None = None,
        qkv_proj_init_fn: Callable[[Linear], None] | None = None,
        q_norm: LayerNorm | None = None,
        k_norm: LayerNorm | None = None,
        rope_module: OLMO2RotaryEmbedding | None = None,
        output_proj: Projection | None = None,
        output_proj_init_fn: Callable[[Linear], None] | None = None,
        bias: bool = True,
        output_proj_bias: bool | None = None,
        state_factory=None,
        gangs: Gangs | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """Initialize OLMO2 Multi-head Attention.

        Args:
            rope_module: adapted from the HuggingFace-style RoPE module for applying rotary position embeddings.
            All other parameters are passed through to StandardMultiheadAttention.
        """
        super().__init__(
            model_dim=model_dim,
            num_heads=num_heads,
            sdpa=sdpa,
            head_dim=head_dim,
            num_key_value_heads=num_key_value_heads,
            kv_dim=kv_dim,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            qkv_proj_init_fn=qkv_proj_init_fn,
            q_norm=q_norm,
            k_norm=k_norm,
            pos_encoder=None,  # We don't use pos_encoder, use rope_module instead
            output_proj=output_proj,
            output_proj_init_fn=output_proj_init_fn,
            bias=bias,
            output_proj_bias=output_proj_bias,
            state_factory=state_factory,
            gangs=gangs,
            device=device,
            dtype=dtype,
        )

        self.rope_module = rope_module

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
        """Forward pass with HuggingFace-style RoPE application.

        This overrides the parent forward to apply RoPE in the HuggingFace style
        (using rotate_half) instead of the complex number approach.
        """
        # Project Q, K, V with normalization
        # Q: (N, S, K_proj) -> norm -> (N, S, H, K_h)
        q = self.q_proj(seqs)
        if self.q_norm is not None:
            q = self.q_norm(q)
        q = q.unflatten(-1, (-1, self.head_dim))

        # K, V: (N, S, K_proj) -> norm -> (N, S, H, K_h)
        k = self.k_proj(keys)
        v = self.v_proj(values)
        if self.k_norm is not None:
            k = self.k_norm(k)
        k = k.unflatten(-1, (-1, self.head_dim))
        v = v.unflatten(-1, (-1, self.head_dim))

        # Apply RoPE using HuggingFace style
        if self.rope_module is not None:
            batch_size, seq_len = seqs.shape[:2]

            # Create position IDs: (B, S)
            position_ids = torch.arange(
                seq_len, device=seqs.device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)

            # Get cos/sin embeddings: (B, S, D)
            cos, sin = self.rope_module(q, position_ids)

            # Transpose Q, K to (B, H, S, D) for RoPE application
            q = q.transpose(1, 2)  # (B, S, H, D) -> (B, H, S, D)
            k = k.transpose(1, 2)  # (B, S, H, D) -> (B, H, S, D)

            # Apply RoPE
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # Transpose back to (B, S, H, D) for SDPA
            q = q.transpose(1, 2)  # (B, H, S, D) -> (B, S, H, D)
            k = k.transpose(1, 2)  # (B, H, S, D) -> (B, S, H, D)

        # Apply SDPA
        # q, k, v are all in (B, S, H, D) format
        # SDPA expects: (q, seqs_layout, k, keys_layout, v, bias_cache)
        needs_weights = len(self._attn_weight_hooks) > 0

        attns, attn_weights = self.sdpa(
            q, seqs_layout, k, keys_layout, v, bias_cache, needs_weights=needs_weights
        )

        if attn_weights is not None:
            for hook in self._attn_weight_hooks.values():
                hook(self, attns, attn_weights)

        # (N, S, H, V_h) -> (N, S, V_proj)
        attns = attns.flatten(-2, -1)

        # Output projection: (N, S, V_proj) -> (N, S, M)
        return self.output_proj(attns)
