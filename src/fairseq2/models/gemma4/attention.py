# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-head attention for Gemma 4 models.

Differs from ``StandardMultiheadAttention`` in several ways:

1. **Partial RoPE** for full (global) attention layers: only the first
   ``encoding_dim`` dimensions of each head are rotated.
2. **K=V mechanism**: when ``k_eq_v=True``, the V projection is omitted and
   the K output (after ``k_norm``) is reused as V (then ``v_norm`` is applied).
3. **V norm**: an optional RMSNorm (without learnable scale) applied to V.
4. **KV sharing**: SOURCE layers expose computed K/V via a callback; CONSUMER
   layers receive pre-computed K/V.
5. **No output gating** (unlike Qwen 3.5).

Reference: Gemma 4 architecture (E4B / 31B / 26B-A4B).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import torch
from torch import Tensor

from fairseq2.models.transformer import (
    SDPA,
    AttentionBiasCache,
    AttentionState,
    AttentionStateFactory,
    FullAttentionState,
    MultiheadAttention,
)
from fairseq2.nn import (
    BatchLayout,
    IncrementalStateBag,
    LayerNorm,
    Linear,
    PositionEncoder,
)
from fairseq2.nn.position_encoder import ReferenceRotaryEncoder
from fairseq2.ops import repeat_interleave


class Gemma4ProportionalRotaryEncoder(ReferenceRotaryEncoder):
    """Rotary position encoder with proportional (partial) rotation.

    Matches the HuggingFace Transformers "proportional" RoPE implementation
    where ``rotate_half`` operates on the full ``head_dim`` tensor.

    Unlike the "split" approach (where only the first ``rope_dim`` dimensions
    are passed through a smaller RoPE encoder and the rest pass through
    unchanged), this encoder produces cos/sin of the **full** ``head_dim``.
    Non-rotary dimensions receive ``cos=1, sin=0`` via zero-padded inverse
    frequencies, so they pass through as an identity transform.

    This is important because ``rotate_half`` pairs dimension *i* with
    dimension *i + head_dim // 2*.  The split approach would pair *i* with
    *i + rope_dim // 2*, producing mathematically different embeddings.

    :param head_dim:
        Full attention head dimension (e.g. 512).
    :param rope_dim:
        Number of dimensions that receive real rotation
        (``head_dim * partial_rotary_factor``, e.g. 128).
    :param max_seq_len:
        Maximum sequence length.
    :param theta:
        RoPE base theta.
    """

    rope_dim: int
    _head_dim: int

    def __init__(
        self,
        head_dim: int,
        rope_dim: int,
        max_seq_len: int,
        *,
        theta: float = 10_000.0,
        device: torch.device | None = None,
    ) -> None:
        self.rope_dim = rope_dim
        self._head_dim = head_dim
        # encoding_dim = head_dim → _rotate_half_way splits at head_dim // 2,
        # matching HF's rotate_half.
        super().__init__(
            encoding_dim=head_dim,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
        )

    def reset_non_persistent_buffers(self) -> None:
        self.cos_freqs[0] = 0.0  # pad
        self.sin_freqs[0] = 0.0  # pad

        device = self.cos_freqs.device
        head_dim = self._head_dim
        rope_angles = self.rope_dim // 2
        nope_angles = head_dim // 2 - rope_angles

        # HF-style frequencies: inv_freq = 1 / (theta^(2i / head_dim))
        # for i in [0, rope_angles).
        indices = torch.arange(rope_angles, device=device, dtype=torch.float32)
        inv_freq_rotated = self.theta ** (-2.0 * indices / head_dim)

        # Zero-pad for NoPE dimensions → cos=1, sin=0 → identity.
        if nope_angles > 0:
            inv_freq = torch.cat(
                [
                    inv_freq_rotated,
                    torch.zeros(
                        nope_angles, device=device, dtype=torch.float32
                    ),
                ],
                dim=0,
            )
        else:
            inv_freq = inv_freq_rotated

        # (S, head_dim // 2)
        steps = torch.arange(
            self.max_seq_len, device=device, dtype=torch.float32
        )
        table = steps.unsqueeze(1) * inv_freq.unsqueeze(0)

        cos = torch.cos(table)
        sin = torch.sin(table)

        # Duplicate halves — matches HF's cat((freqs, freqs), dim=-1).
        self.cos_freqs[1:, : head_dim // 2] = cos
        self.cos_freqs[1:, head_dim // 2 :] = cos
        self.sin_freqs[1:, : head_dim // 2] = sin
        self.sin_freqs[1:, head_dim // 2 :] = sin


class Gemma4Attention(MultiheadAttention):
    """Multi-head attention for Gemma 4 decoder layers.

    Key features:

    * **Partial RoPE** --- when ``pos_encoder.encoding_dim < head_dim``, only
      the first ``encoding_dim`` dimensions are rotated and the remainder pass
      through unchanged.  Full (global) attention layers typically use
      ``global_head_dim=512`` with ``partial_rotary_factor=0.25`` so that 128
      dimensions are rotated.  Sliding (local) attention layers rotate all 256
      dimensions.
    * **K=V** --- when *k_eq_v* is ``True`` the constructor does **not** create
      a V projection.  Instead, the K output (after ``k_norm``) is reused as V
      and then ``v_norm`` is applied.
    * **V norm** --- an optional :class:`LayerNorm` (typically RMSNorm with
      ``elementwise_affine=False``) applied to V after projection (or after K
      reuse).
    * **KV sharing** --- SOURCE layers store K/V via *kv_storage_callback*;
      CONSUMER layers receive pre-computed K/V via *pre_computed_kv*.
    * **QK-Norm** --- per-head ``q_norm`` / ``k_norm`` applied after unflatten.
    """

    num_heads: Final[int]
    num_key_value_heads: Final[int]
    num_query_groups: Final[int]
    head_dim: Final[int]
    k_eq_v: Final[bool]
    is_kv_consumer: Final[bool]

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
        v_norm: LayerNorm | None = None,
        k_eq_v: bool = False,
        is_kv_consumer: bool = False,
        state_factory: AttentionStateFactory | None = None,
        qkv_proj_init_fn: Callable[[Linear], None] | None = None,
        output_proj_init_fn: Callable[[Linear], None] | None = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param num_heads:
            The number of query attention heads.
        :param sdpa:
            The scaled dot-product attention module.
        :param head_dim:
            The dimensionality of each attention head.
        :param num_key_value_heads:
            The number of key/value heads for Grouped Query Attention.  If
            ``None``, defaults to *num_heads* (standard MHA).
        :param pos_encoder:
            Position encoder (typically RoPE).  When its ``encoding_dim`` is
            smaller than *head_dim*, partial rotation is applied.
        :param q_norm:
            Layer norm applied to queries after unflatten.
        :param k_norm:
            Layer norm applied to keys after unflatten.
        :param v_norm:
            Layer norm applied to values (typically RMSNorm without learnable
            scale).
        :param k_eq_v:
            If ``True``, skip the V projection and reuse K output as V.
        :param is_kv_consumer:
            If ``True``, this layer receives pre-computed K/V from a SOURCE
            layer via KV sharing.  K/V projections and k_norm are **not**
            created (matching HuggingFace, which also omits these for
            consumer layers).
        :param state_factory:
            Factory for :class:`AttentionState` (incremental decoding cache).
        :param qkv_proj_init_fn:
            Custom initializer for Q/K/V projection weights.
        :param output_proj_init_fn:
            Custom initializer for the output projection weights.
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.k_eq_v = k_eq_v
        self.is_kv_consumer = is_kv_consumer

        if num_key_value_heads is None:
            num_key_value_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_query_groups = num_heads // num_key_value_heads

        # -- Projections --
        self.q_proj = Linear(
            model_dim,
            num_heads * head_dim,
            bias=False,
            init_fn=qkv_proj_init_fn,
        )

        # K/V projections are only created for non-consumer layers.
        # Consumer layers receive pre-computed K/V from SOURCE layers
        # via KV sharing and do not need their own projections.
        if not is_kv_consumer:
            self.k_proj = Linear(
                model_dim,
                num_key_value_heads * head_dim,
                bias=False,
                init_fn=qkv_proj_init_fn,
            )

            # V projection is only created when k_eq_v is False.
            if not k_eq_v:
                self.v_proj = Linear(
                    model_dim,
                    num_key_value_heads * head_dim,
                    bias=False,
                    init_fn=qkv_proj_init_fn,
                )

        self.output_proj = Linear(
            num_heads * head_dim,
            model_dim,
            bias=False,
            init_fn=output_proj_init_fn,
        )

        # -- Norms --
        self.q_norm = q_norm
        # k_norm is only set for non-consumer layers.
        self.k_norm = None if is_kv_consumer else k_norm
        self.v_norm = None if is_kv_consumer else v_norm

        # -- Position encoder & SDPA --
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
        pre_computed_kv: tuple[Tensor, Tensor] | None = None,
        kv_storage_callback: Callable[[Tensor, Tensor], None] | None = None,
    ) -> Tensor:
        """
        :param seqs:
            The query sequences. *Shape:* ``(B, S, model_dim)``.
        :param seqs_layout:
            Batch layout for *seqs*.
        :param keys:
            The key sequences (typically same as *seqs* for self-attention).
        :param keys_layout:
            Batch layout for *keys*.
        :param values:
            The value sequences (typically same as *seqs* for self-attention).
        :param bias_cache:
            Attention bias cache.
        :param state_bag:
            Incremental state bag for decoding.
        :param pre_computed_kv:
            Pre-computed ``(K, V)`` tensors from a SOURCE layer.  When provided,
            K/V projection and RoPE are skipped (CONSUMER path).
        :param kv_storage_callback:
            Callback invoked with ``(K, V)`` after computation so that a SOURCE
            layer can store them for downstream CONSUMERs.

        :returns:
            The attention output. *Shape:* ``(B, S, model_dim)``.
        """
        # ---- Q projection ----
        # (B, S, model_dim) -> (B, S, num_heads * head_dim)
        q = self.q_proj(seqs)
        # (B, S, num_heads * head_dim) -> (B, S, num_heads, head_dim)
        q = q.unflatten(-1, (self.num_heads, self.head_dim))

        # ---- Q norm ----
        if self.q_norm is not None:
            q = self.q_norm(q)

        # ---- Q RoPE (applied unconditionally — even for CONSUMER layers) ----
        # HF always applies RoPE to Q.  Only K skips RoPE in the consumer
        # path (since K was already RoPE'd by the SOURCE layer).
        if self.pos_encoder is not None:
            q = self._apply_rope(q, seqs_layout, state_bag)

        # ---- K/V path ----
        if pre_computed_kv is not None:
            # CONSUMER path: use pre-computed K/V from a SOURCE layer.
            k, v = pre_computed_kv
        elif self.is_kv_consumer:
            raise RuntimeError(
                "KV-consumer layer called without pre_computed_kv.  "
                "Consumer layers do not have k_proj/v_proj."
            )
        else:
            # ---- K projection ----
            # (B, S, model_dim) -> (B, S, num_kv_heads * head_dim)
            k = self.k_proj(keys)
            # (B, S, num_kv_heads * head_dim) -> (B, S, num_kv_heads, head_dim)
            k = k.unflatten(-1, (self.num_key_value_heads, self.head_dim))

            # ---- V projection / K=V ----
            # IMPORTANT: When k_eq_v=True, V gets the raw K projection output
            # BEFORE k_norm.  HF does:
            #   value_states = key_states  (before k_norm)
            #   key_states = k_norm(key_states)
            #   value_states = v_norm(value_states)
            # So v_norm sees the unnormalized projection, not the k_norm'd one.
            if self.k_eq_v:
                v = k  # Raw k_proj output, before k_norm.
            else:
                # (B, S, model_dim) -> (B, S, num_kv_heads, head_dim)
                v = self.v_proj(values)
                v = v.unflatten(-1, (self.num_key_value_heads, self.head_dim))

            # ---- K norm (applied AFTER saving raw K for V when k_eq_v) ----
            if self.k_norm is not None:
                k = self.k_norm(k)

            # ---- V norm ----
            if self.v_norm is not None:
                v = self.v_norm(v)

            # ---- K RoPE (only for non-consumer layers) ----
            if self.pos_encoder is not None:
                k = self._apply_rope(k, keys_layout, state_bag)

        # ---- KV cache management ----
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

        # ---- Store K/V for downstream consumers (SOURCE path) ----
        if kv_storage_callback is not None:
            kv_storage_callback(k, v)

        # ---- GQA expansion ----
        if self.num_query_groups > 1:
            k = repeat_interleave(k, dim=-2, repeat=self.num_query_groups)
            v = repeat_interleave(v, dim=-2, repeat=self.num_query_groups)

        # ---- Scaled dot-product attention ----
        # q, k, v: (B, S, H, D)
        attn_output, _ = self.sdpa(
            q, seqs_layout, k, keys_layout, v, bias_cache
        )

        # ---- Output: flatten heads and project ----
        # (B, S, H, D) -> (B, S, H * D)
        attn_output = attn_output.flatten(-2)

        # (B, S, H * D) -> (B, S, model_dim)
        return self.output_proj(attn_output)

    def _apply_rope(
        self,
        x: Tensor,
        layout: BatchLayout,
        state_bag: IncrementalStateBag | None,
    ) -> Tensor:
        """Apply RoPE to ``x`` using ``self.pos_encoder``.

        Gemma 4 always uses a position encoder whose ``encoding_dim`` equals
        ``head_dim``.  For partial RoPE on global attention layers, use
        :class:`Gemma4ProportionalRotaryEncoder`, which keeps ``encoding_dim
        == head_dim`` and zero-pads ``inv_freq`` so the non-rotary dimensions
        act as identity (``cos=1, sin=0``).  This matches HuggingFace's
        ``rotate_half`` pairing.
        """
        assert self.pos_encoder is not None
        assert self.pos_encoder.encoding_dim == self.head_dim, (
            f"Gemma4Attention expects pos_encoder.encoding_dim "
            f"({self.pos_encoder.encoding_dim}) == head_dim ({self.head_dim}). "
            f"For partial RoPE use Gemma4ProportionalRotaryEncoder."
        )
        return self.pos_encoder(x, layout, state_bag=state_bag)
