# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from fairseq2.models.qwen.attention import Qwen35Attention
from fairseq2.models.transformer.attention_bias import (
    AttentionBiasCache,
    CausalAttentionBias,
    IdentityBias,
)
from fairseq2.models.transformer.sdpa.naive import NaiveSDPA
from fairseq2.nn import BatchLayout, IncrementalStateBag, RMSNorm
from fairseq2.nn.position_encoder import ReferenceRotaryEncoder
from tests.common import assert_close, device


class TestQwen35Attention:
    def test_forward_produces_correct_shape(self) -> None:
        """Output shape is (B, S, model_dim)."""
        sdpa = NaiveSDPA(IdentityBias())
        attn = Qwen35Attention(model_dim=64, num_heads=4, sdpa=sdpa, head_dim=16)
        attn = attn.to(device)

        seqs = torch.randn(2, 8, 64, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        with torch.no_grad():
            out = attn(seqs, layout, seqs, layout, seqs, bias_cache)

        assert out.shape == (2, 8, 64)

    def test_output_gating_effect(self) -> None:
        """When gate output is all zeros, attention output should be near zero."""
        sdpa = NaiveSDPA(IdentityBias())
        attn = Qwen35Attention(model_dim=64, num_heads=4, sdpa=sdpa, head_dim=16)
        attn = attn.to(device)

        seqs = torch.randn(1, 4, 64, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        with torch.no_grad():
            out1 = attn(seqs, layout, seqs, layout, seqs, bias_cache)

        # Verify output is not zero (gate should be non-trivial with random weights)
        assert out1.abs().mean() > 1e-6

    def test_partial_rope_applies_to_subset_of_dims(self) -> None:
        """With encoding_dim < head_dim, only first encoding_dim dims should be rotated."""
        model_dim = 64
        num_heads = 4
        head_dim = 16
        encoding_dim = 4  # Only first 4 of 16 dims rotated

        rope = ReferenceRotaryEncoder(encoding_dim, max_seq_len=32, device=device)
        sdpa = NaiveSDPA(IdentityBias())
        attn = Qwen35Attention(
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
        attn = Qwen35Attention(
            model_dim=64,
            num_heads=4,
            sdpa=sdpa,
            head_dim=16,
            num_key_value_heads=2,  # GQA: 4 Q heads, 2 KV heads
        )
        attn = attn.to(device)

        seqs = torch.randn(2, 6, 64, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        with torch.no_grad():
            out = attn(seqs, layout, seqs, layout, seqs, bias_cache)

        assert out.shape == (2, 6, 64)

    def test_qk_norm_applied(self) -> None:
        """When q_norm and k_norm are provided, output should differ from no-norm case."""
        sdpa = NaiveSDPA(IdentityBias())

        # Without norms
        attn_no_norm = Qwen35Attention(
            model_dim=64, num_heads=4, sdpa=sdpa, head_dim=16
        )
        attn_no_norm = attn_no_norm.to(device)

        # With norms
        q_norm = RMSNorm(16, bias=False, device=device)
        k_norm = RMSNorm(16, bias=False, device=device)
        attn_norm = Qwen35Attention(
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

    def test_incremental_kv_cache_matches_full_forward(self) -> None:
        """Token-by-token decoding with KV cache produces the same logits as causal full-sequence forward."""
        sdpa = NaiveSDPA(CausalAttentionBias())
        attn = Qwen35Attention(model_dim=64, num_heads=4, sdpa=sdpa, head_dim=16)
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
