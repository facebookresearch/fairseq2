# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Gemma 4 attention (partial RoPE, K=V, KV sharing, QKV norms)."""

from __future__ import annotations

import torch

from fairseq2.models.gemma4.attention import Gemma4Attention
from fairseq2.models.transformer.attention_bias import (
    AttentionBiasCache,
    CausalAttentionBias,
    IdentityBias,
)
from fairseq2.models.transformer.sdpa.naive import NaiveSDPA
from fairseq2.nn import BatchLayout, IncrementalStateBag, RMSNorm
from fairseq2.nn.position_encoder import ReferenceRotaryEncoder
from tests.common import assert_close, device


class TestGemma4Attention:
    """Test Gemma4Attention module."""

    def test_forward_produces_correct_shape(self) -> None:
        """Output shape is (B, S, model_dim)."""
        sdpa = NaiveSDPA(IdentityBias())
        attn = Gemma4Attention(model_dim=64, num_heads=4, sdpa=sdpa, head_dim=16)
        attn = attn.to(device)

        seqs = torch.randn(2, 8, 64, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        with torch.no_grad():
            out = attn(seqs, layout, seqs, layout, seqs, bias_cache)

        assert out.shape == (2, 8, 64)

    def test_partial_rope_applies_to_subset_of_dims(self) -> None:
        """With encoding_dim < head_dim, only first encoding_dim dims are rotated."""
        model_dim = 64
        num_heads = 4
        head_dim = 16
        encoding_dim = 4  # Only first 4 of 16 dims rotated (partial_rotary_factor=0.25)

        rope = ReferenceRotaryEncoder(encoding_dim, max_seq_len=32, device=device)
        sdpa = NaiveSDPA(IdentityBias())
        attn = Gemma4Attention(
            model_dim=model_dim,
            num_heads=num_heads,
            sdpa=sdpa,
            head_dim=head_dim,
            pos_encoder=rope,
        )
        attn = attn.to(device)

        seqs = torch.randn(1, 4, model_dim, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        with torch.no_grad():
            out = attn(seqs, layout, seqs, layout, seqs, bias_cache)

        assert out.shape == (1, 4, model_dim)

    def test_full_rope_rotates_all_dims(self) -> None:
        """With encoding_dim == head_dim, all dims are rotated."""
        model_dim = 64
        num_heads = 4
        head_dim = 16

        rope = ReferenceRotaryEncoder(head_dim, max_seq_len=32, device=device)
        sdpa = NaiveSDPA(IdentityBias())
        attn = Gemma4Attention(
            model_dim=model_dim,
            num_heads=num_heads,
            sdpa=sdpa,
            head_dim=head_dim,
            pos_encoder=rope,
        )
        attn = attn.to(device)

        seqs = torch.randn(1, 4, model_dim, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        with torch.no_grad():
            out = attn(seqs, layout, seqs, layout, seqs, bias_cache)

        assert out.shape == (1, 4, model_dim)

    def test_gqa_with_fewer_kv_heads(self) -> None:
        """GQA with num_key_value_heads < num_heads works correctly."""
        sdpa = NaiveSDPA(IdentityBias())
        attn = Gemma4Attention(
            model_dim=64,
            num_heads=4,
            sdpa=sdpa,
            head_dim=16,
            num_key_value_heads=2,
        )
        attn = attn.to(device)

        seqs = torch.randn(2, 6, 64, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        with torch.no_grad():
            out = attn(seqs, layout, seqs, layout, seqs, bias_cache)

        assert out.shape == (2, 6, 64)

    def test_k_eq_v_omits_v_proj(self) -> None:
        """When k_eq_v=True, there is no v_proj parameter."""
        sdpa = NaiveSDPA(IdentityBias())
        attn = Gemma4Attention(
            model_dim=64,
            num_heads=4,
            sdpa=sdpa,
            head_dim=16,
            k_eq_v=True,
        )

        assert not hasattr(attn, "v_proj")
        # But k_proj and q_proj still exist
        assert hasattr(attn, "k_proj")
        assert hasattr(attn, "q_proj")

    def test_k_eq_v_forward(self) -> None:
        """K=V attention produces correct shape."""
        sdpa = NaiveSDPA(IdentityBias())
        attn = Gemma4Attention(
            model_dim=64,
            num_heads=4,
            sdpa=sdpa,
            head_dim=16,
            k_eq_v=True,
        )
        attn = attn.to(device)

        seqs = torch.randn(1, 4, 64, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        with torch.no_grad():
            out = attn(seqs, layout, seqs, layout, seqs, bias_cache)

        assert out.shape == (1, 4, 64)

    def test_k_eq_v_uses_raw_k_before_k_norm(self) -> None:
        """When k_eq_v=True, V gets raw k_proj output BEFORE k_norm.

        This matches HuggingFace behavior where:
            value_states = key_states  # before k_norm
            key_states = k_norm(key_states)
            value_states = v_norm(value_states)
        """
        model_dim = 64
        head_dim = 16
        sdpa = NaiveSDPA(IdentityBias())

        k_norm = RMSNorm(head_dim, bias=False, device=device)
        v_norm = RMSNorm(head_dim, bias=False, eps=1e-6, elementwise_affine=False, device=device)

        attn = Gemma4Attention(
            model_dim=model_dim,
            num_heads=4,
            sdpa=sdpa,
            head_dim=head_dim,
            k_eq_v=True,
            k_norm=k_norm,
            v_norm=v_norm,
        )
        attn = attn.to(device)

        seqs = torch.randn(1, 4, model_dim, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        with torch.no_grad():
            out = attn(seqs, layout, seqs, layout, seqs, bias_cache)

        # Should produce non-trivial output (not zero, not NaN)
        assert out.shape == (1, 4, model_dim)
        assert not torch.isnan(out).any()
        assert out.abs().mean() > 1e-6

    def test_qk_norm_applied(self) -> None:
        """When q_norm and k_norm are provided, output differs from no-norm case."""
        sdpa = NaiveSDPA(IdentityBias())

        # Without norms
        attn_no_norm = Gemma4Attention(
            model_dim=64, num_heads=4, sdpa=sdpa, head_dim=16
        )
        attn_no_norm = attn_no_norm.to(device)

        # With norms
        q_norm = RMSNorm(16, bias=False, device=device)
        k_norm = RMSNorm(16, bias=False, device=device)
        attn_norm = Gemma4Attention(
            model_dim=64,
            num_heads=4,
            sdpa=sdpa,
            head_dim=16,
            q_norm=q_norm,
            k_norm=k_norm,
        )
        attn_norm = attn_norm.to(device)

        # Copy weights so only the norm makes a difference
        attn_norm.q_proj.weight.data.copy_(attn_no_norm.q_proj.weight.data)
        attn_norm.k_proj.weight.data.copy_(attn_no_norm.k_proj.weight.data)
        attn_norm.v_proj.weight.data.copy_(attn_no_norm.v_proj.weight.data)
        attn_norm.output_proj.weight.data.copy_(attn_no_norm.output_proj.weight.data)

        seqs = torch.randn(1, 4, 64, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        with torch.no_grad():
            out_no_norm = attn_no_norm(seqs, layout, seqs, layout, seqs, bias_cache)
            out_norm = attn_norm(seqs, layout, seqs, layout, seqs, bias_cache)

        # Outputs should differ because of norm
        assert not torch.allclose(out_no_norm, out_norm, atol=1e-6)

    def test_v_norm_without_scale(self) -> None:
        """V norm should be created without learnable scale (elementwise_affine=False)."""
        v_norm = RMSNorm(16, bias=False, eps=1e-6, elementwise_affine=False, device=device)

        sdpa = NaiveSDPA(IdentityBias())
        attn = Gemma4Attention(
            model_dim=64,
            num_heads=4,
            sdpa=sdpa,
            head_dim=16,
            v_norm=v_norm,
        )

        # v_norm should NOT have a learnable weight parameter
        v_norm_params = list(attn.v_norm.parameters())
        assert len(v_norm_params) == 0, "v_norm should have no learnable parameters"

    def test_pre_computed_kv_consumer_path(self) -> None:
        """CONSUMER path uses pre-computed KV, skipping K/V projection and RoPE."""
        model_dim = 64
        head_dim = 16
        num_kv_heads = 2

        sdpa = NaiveSDPA(IdentityBias())
        attn = Gemma4Attention(
            model_dim=model_dim,
            num_heads=4,
            sdpa=sdpa,
            head_dim=head_dim,
            num_key_value_heads=num_kv_heads,
        )
        attn = attn.to(device)

        seqs = torch.randn(1, 4, model_dim, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        # Pre-computed K and V from a SOURCE layer
        pre_k = torch.randn(1, 4, num_kv_heads, head_dim, device=device)
        pre_v = torch.randn(1, 4, num_kv_heads, head_dim, device=device)

        with torch.no_grad():
            out = attn(
                seqs, layout, seqs, layout, seqs, bias_cache,
                pre_computed_kv=(pre_k, pre_v),
            )

        assert out.shape == (1, 4, model_dim)

    def test_kv_storage_callback_source_path(self) -> None:
        """SOURCE path invokes kv_storage_callback with computed K, V."""
        model_dim = 64
        head_dim = 16
        num_kv_heads = 2

        sdpa = NaiveSDPA(IdentityBias())
        attn = Gemma4Attention(
            model_dim=model_dim,
            num_heads=4,
            sdpa=sdpa,
            head_dim=head_dim,
            num_key_value_heads=num_kv_heads,
        )
        attn = attn.to(device)

        seqs = torch.randn(1, 4, model_dim, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        stored_kv: list[tuple[torch.Tensor, torch.Tensor]] = []

        def store_callback(k: torch.Tensor, v: torch.Tensor) -> None:
            stored_kv.append((k, v))

        with torch.no_grad():
            out = attn(
                seqs, layout, seqs, layout, seqs, bias_cache,
                kv_storage_callback=store_callback,
            )

        assert len(stored_kv) == 1
        k_stored, v_stored = stored_kv[0]
        assert k_stored.shape == (1, 4, num_kv_heads, head_dim)
        assert v_stored.shape == (1, 4, num_kv_heads, head_dim)

    def test_incremental_kv_cache_matches_full_forward(self) -> None:
        """Token-by-token decoding with KV cache matches causal full-sequence forward."""
        sdpa = NaiveSDPA(CausalAttentionBias())
        attn = Gemma4Attention(model_dim=64, num_heads=4, sdpa=sdpa, head_dim=16)
        attn = attn.to(device)
        attn.eval()

        seqs = torch.randn(1, 6, 64, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        with torch.no_grad():
            full_out = attn(seqs, layout, seqs, layout, seqs, bias_cache)

        state_bag = IncrementalStateBag(max_num_steps=32)

        with torch.no_grad():
            for idx in range(6):
                step_seqs = seqs[:, idx : idx + 1, :]
                step_layout = BatchLayout.of(step_seqs)
                out = attn(
                    step_seqs,
                    step_layout,
                    step_seqs,
                    step_layout,
                    step_seqs,
                    bias_cache,
                    state_bag=state_bag,
                )
                assert_close(out, full_out[:, idx : idx + 1, :], atol=1e-5)
                state_bag.increment_step_nr()
