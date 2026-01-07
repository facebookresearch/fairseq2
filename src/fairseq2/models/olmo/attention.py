# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OLMO-specific attention module with Q/K normalization."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

from fairseq2.gang import Gangs
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


class OLMOMultiheadAttention(StandardMultiheadAttention):
    """OLMO Multi-head Attention with Q/K normalization and reference rotary encoding."""

    rope_encoder: PositionEncoder | None

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
        rope_encoder: PositionEncoder | None = None,
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
            rope_encoder: rotary encoder applied to queries and keys after projection.
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
            pos_encoder=None,  
            output_proj=output_proj,
            output_proj_init_fn=output_proj_init_fn,
            bias=bias,
            output_proj_bias=output_proj_bias,
            state_factory=state_factory,
            gangs=gangs,
            device=device,
            dtype=dtype,
        )

        self.rope_encoder = rope_encoder

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
        """Forward pass with OLMO2 rotary encoding semantics."""
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

        # Apply rotary encoding via the shared encoder.
        rope_encoder = self.rope_encoder
        if rope_encoder is not None:
            q = rope_encoder(q, seqs_layout, state_bag=state_bag)
            k = rope_encoder(k, keys_layout, state_bag=state_bag)

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
