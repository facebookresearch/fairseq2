# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OLMO-specific attention module with Q/K normalization."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor
from typing_extensions import final, override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.gang import Gangs
from fairseq2.models.transformer.attention_bias import AttentionBiasCache
from fairseq2.models.transformer.multihead_attention import (
    AttentionState,
    AttentionStateFactory,
    FullAttentionState,
    MultiheadAttention,
    init_mha_output_projection,
    init_qkv_projection,
)
from fairseq2.models.transformer.sdpa.base import SDPA
from fairseq2.nn import (
    BatchLayout,
    ColumnShardedLinear,
    IncrementalStateBag,
    LayerNorm,
    Linear,
    PositionEncoder,
    Projection,
    RowShardedLinear,
)
from fairseq2.nn.utils.module import get_name_or_self
from fairseq2.ops import repeat_interleave


@final
class OLMOMultiheadAttention(MultiheadAttention):
    """OLMO Multi-head Attention with Q/K normalization and rotary encoding.

    This class inherits directly from :class:`MultiheadAttention` (the abstract
    base) and inlines the projection/norm setup that
    :class:`StandardMultiheadAttention` provides, but replaces the
    ``pos_encoder`` parameter with ``rope_encoder`` (OLMO-style RoPE).

    During incremental decoding (``not self.training`` and ``state_bag`` is
    provided), projected keys and values are cached via :class:`AttentionState`.
    The cache implementation is determined by ``state_factory``; when
    ``state_factory`` is ``None`` it defaults to :class:`FullAttentionState`.
    For sliding-window layers, pass a :class:`LocalAttentionStateFactory`.

    .. note::
        This module only supports self-attention (decoder-only). Cross-attention
        (encoder-decoder) is not supported since OLMO is a decoder-only model.
    """

    rope_encoder: PositionEncoder | None

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        sdpa: SDPA,
        *,
        head_dim: int | None = None,
        num_key_value_heads: int | None = None,
        qkv_proj_init_fn: Callable[[Linear], None] | None = None,
        q_norm: LayerNorm | None = None,
        k_norm: LayerNorm | None = None,
        rope_encoder: PositionEncoder | None = None,
        output_proj_init_fn: Callable[[Linear], None] | None = None,
        bias: bool = True,
        output_proj_bias: bool | None = None,
        state_factory: AttentionStateFactory | None = None,
        gangs: Gangs | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        # --- num_heads / num_key_value_heads validation ---
        self.num_heads = num_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_heads
        else:
            if num_heads < num_key_value_heads:
                raise ValueError(
                    f"`num_heads` must be greater than or equal to "
                    f"`num_key_value_heads` ({num_key_value_heads}), "
                    f"but is {num_heads} instead."
                )
            if num_heads % num_key_value_heads != 0:
                raise ValueError(
                    f"`num_heads` must be a multiple of "
                    f"`num_key_value_heads` ({num_key_value_heads}), "
                    f"but is {num_heads} instead."
                )

        self.num_key_value_heads = num_key_value_heads

        if head_dim is None:
            head_dim = model_dim // num_heads

        self.head_dim = head_dim

        self.num_query_groups = num_heads // num_key_value_heads

        # --- Q / K / V projections ---
        self.q_proj: Projection = ColumnShardedLinear(
            model_dim,
            head_dim * num_heads,
            bias,
            gather_output=False,
            init_fn=qkv_proj_init_fn or init_qkv_projection,
            gangs=gangs,
            device=device,
            dtype=dtype,
        )
        self.k_proj: Projection = ColumnShardedLinear(
            model_dim,
            head_dim * num_key_value_heads,
            bias,
            gather_output=False,
            init_fn=qkv_proj_init_fn or init_qkv_projection,
            gangs=gangs,
            device=device,
            dtype=dtype,
        )
        self.v_proj: Projection = ColumnShardedLinear(
            model_dim,
            head_dim * num_key_value_heads,
            bias,
            gather_output=False,
            init_fn=qkv_proj_init_fn or init_qkv_projection,
            gangs=gangs,
            device=device,
            dtype=dtype,
        )

        # --- Q / K norms ---
        self.q_norm: LayerNorm | None
        self.k_norm: LayerNorm | None

        self.register_module("q_norm", q_norm)
        self.register_module("k_norm", k_norm)

        # --- Rope encoder ---
        if rope_encoder is not None:
            if head_dim != rope_encoder.encoding_dim:
                raise ValueError(
                    f"`rope_encoder.encoding_dim` must be equal to the size "
                    f"of the head dimension ({head_dim}), but is "
                    f"{rope_encoder.encoding_dim} instead."
                )

        self.rope_encoder = rope_encoder

        # --- SDPA ---
        self.sdpa = sdpa

        # --- Output projection ---
        v_dim = self.v_proj.output_dim * self.num_query_groups

        if output_proj_bias is None:
            output_proj_bias = bias

        self.output_proj: Projection = RowShardedLinear(
            v_dim,
            model_dim,
            output_proj_bias,
            scatter_input=False,
            init_fn=output_proj_init_fn or init_mha_output_projection,
            gangs=gangs,
            device=device,
            dtype=dtype,
        )

        self.state_factory = state_factory

    @override
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

        # KV caching for incremental (auto-regressive) decoding.
        if not self.training and state_bag is not None and seqs is keys:
            if keys_layout.packed:
                raise ValueError("`keys` must not be a packed batch.")
            if keys_layout.padded:
                raise ValueError("`keys` must not be a padded batch.")

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

        # With Grouped Query Attention, each key/value head is repeated.
        if self.num_query_groups > 1:
            k = repeat_interleave(k, dim=-2, repeat=self.num_query_groups)
            v = repeat_interleave(v, dim=-2, repeat=self.num_query_groups)

        # Apply SDPA
        needs_weights = len(self._attn_weight_hooks) > 0

        attns, attn_weights = self.sdpa(
            q, seqs_layout, k, keys_layout, v, bias_cache, needs_weights=needs_weights
        )

        del q, k, v

        if attn_weights is not None:
            for hook in self._attn_weight_hooks.values():
                hook(self, attns, attn_weights)

        # (N, S, H, V_h) -> (N, S, V_proj)
        attns = attns.flatten(-2, -1)

        # Output projection: (N, S, V_proj) -> (N, S, M)
        return self.output_proj(attns)

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        s = f"num_heads={self.num_heads}"

        if self.num_key_value_heads != self.num_heads:
            s = f"{s}, num_key_value_heads={self.num_key_value_heads}"

        if self.num_query_groups > 1:
            s = f"{s}, num_query_groups={self.num_query_groups}"

        if self.state_factory is not None:
            state_factory = get_name_or_self(self.state_factory)

            s = f"{s}, state_factory={state_factory}"

        return s
