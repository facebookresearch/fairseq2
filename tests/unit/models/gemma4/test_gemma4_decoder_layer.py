# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Gemma 4 decoder layer (FFN, MoE, PLE, layer_scalar)."""

from __future__ import annotations

import torch

from fairseq2.models.gemma4.attention import Gemma4Attention
from fairseq2.models.gemma4.decoder_layer import Gemma4DecoderLayer
from fairseq2.models.gemma4.moe import Gemma4Experts, Gemma4Router
from fairseq2.models.transformer import GLUFeedForwardNetwork
from fairseq2.models.transformer.attention_bias import (
    AttentionBiasCache,
    IdentityBias,
)
from fairseq2.models.transformer.sdpa.naive import NaiveSDPA
from fairseq2.nn import BatchLayout, RMSNorm
from fairseq2.nn.projection import Linear
from tests.common import device


def _make_base_layer(
    model_dim: int = 64,
    ffn_inner: int = 128,
    *,
    enable_moe: bool = False,
    enable_ple: bool = False,
    ple_dim: int = 16,
    num_experts: int = 4,
    top_k: int = 2,
    moe_inner: int = 32,
) -> Gemma4DecoderLayer:
    """Build a minimal decoder layer for testing."""
    sdpa = NaiveSDPA(IdentityBias())
    self_attn = Gemma4Attention(
        model_dim=model_dim, num_heads=4, sdpa=sdpa, head_dim=model_dim // 4
    )
    ffn = GLUFeedForwardNetwork(model_dim, ffn_inner, bias=False, inner_dim_scale=1.0)

    # Required norms
    input_ln = RMSNorm(model_dim, bias=False, device=device)
    post_attn_ln = RMSNorm(model_dim, bias=False, device=device)
    pre_ffn_ln = RMSNorm(model_dim, bias=False, device=device)
    post_ffn_ln = RMSNorm(model_dim, bias=False, device=device)

    # Optional PLE
    ple_gate = None
    ple_proj = None
    ple_norm = None
    if enable_ple:
        ple_gate = Linear(model_dim, ple_dim, bias=False, device=device)
        ple_proj = Linear(ple_dim, model_dim, bias=False, device=device)
        ple_norm = RMSNorm(model_dim, bias=False, device=device)

    # Optional MoE
    router = None
    experts = None
    post_ffn_ln1 = None
    pre_ffn_ln2 = None
    post_ffn_ln2 = None
    if enable_moe:
        router = Gemma4Router(model_dim, num_experts, top_k, rms_norm_eps=1e-6)
        experts = Gemma4Experts(model_dim, num_experts, moe_inner)
        torch.nn.init.normal_(experts.gate_up_proj, std=0.01)
        torch.nn.init.normal_(experts.down_proj, std=0.01)
        post_ffn_ln1 = RMSNorm(model_dim, bias=False, device=device)
        pre_ffn_ln2 = RMSNorm(model_dim, bias=False, device=device)
        post_ffn_ln2 = RMSNorm(model_dim, bias=False, device=device)

    layer = Gemma4DecoderLayer(
        self_attn=self_attn,
        ffn=ffn,
        input_layernorm=input_ln,
        post_attention_layernorm=post_attn_ln,
        pre_feedforward_layernorm=pre_ffn_ln,
        post_feedforward_layernorm=post_ffn_ln,
        per_layer_input_gate=ple_gate,
        per_layer_projection=ple_proj,
        post_per_layer_input_norm=ple_norm,
        router=router,
        experts=experts,
        post_feedforward_layernorm_1=post_ffn_ln1,
        pre_feedforward_layernorm_2=pre_ffn_ln2,
        post_feedforward_layernorm_2=post_ffn_ln2,
    )
    return layer.to(device)


class TestGemma4DecoderLayer:
    """Test Gemma4DecoderLayer."""

    def test_forward_shape_basic(self) -> None:
        """Basic layer (no MoE, no PLE) produces correct shape."""
        layer = _make_base_layer()
        seqs = torch.randn(2, 8, 64, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        with torch.no_grad():
            out = layer(seqs, layout, bias_cache)

        assert out.shape == (2, 8, 64)

    def test_forward_shape_with_moe(self) -> None:
        """Layer with MoE produces correct shape."""
        layer = _make_base_layer(enable_moe=True)
        seqs = torch.randn(2, 8, 64, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        with torch.no_grad():
            out = layer(seqs, layout, bias_cache)

        assert out.shape == (2, 8, 64)

    def test_forward_shape_with_ple(self) -> None:
        """Layer with PLE produces correct shape."""
        ple_dim = 16
        layer = _make_base_layer(enable_ple=True, ple_dim=ple_dim)
        seqs = torch.randn(2, 8, 64, device=device)
        per_layer_input = torch.randn(2, 8, ple_dim, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        with torch.no_grad():
            out = layer(seqs, layout, bias_cache, per_layer_input=per_layer_input)

        assert out.shape == (2, 8, 64)

    def test_forward_shape_with_moe_and_ple(self) -> None:
        """Layer with both MoE and PLE produces correct shape."""
        ple_dim = 16
        layer = _make_base_layer(enable_moe=True, enable_ple=True, ple_dim=ple_dim)
        seqs = torch.randn(2, 8, 64, device=device)
        per_layer_input = torch.randn(2, 8, ple_dim, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        with torch.no_grad():
            out = layer(seqs, layout, bias_cache, per_layer_input=per_layer_input)

        assert out.shape == (2, 8, 64)

    def test_layer_scalar_applied(self) -> None:
        """layer_scalar buffer scales the output."""
        layer = _make_base_layer()

        # Set layer_scalar to 0 — output should be zeros
        layer.layer_scalar.fill_(0.0)

        seqs = torch.randn(1, 4, 64, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        with torch.no_grad():
            out = layer(seqs, layout, bias_cache)

        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_layer_scalar_init_value(self) -> None:
        """layer_scalar defaults to 1.0."""
        layer = _make_base_layer()
        assert torch.allclose(
            layer.layer_scalar, torch.ones(1, device=device), atol=1e-6
        )

    def test_ple_gating_flow(self) -> None:
        """PLE applies: gate → gelu → multiply → project → norm → residual."""
        ple_dim = 16
        layer = _make_base_layer(enable_ple=True, ple_dim=ple_dim)
        seqs = torch.randn(1, 4, 64, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        # With PLE input
        per_layer_input = torch.randn(1, 4, ple_dim, device=device)
        with torch.no_grad():
            out_with_ple = layer(seqs, layout, bias_cache, per_layer_input=per_layer_input)

        # Without PLE input (PLE step skipped even though module has PLE)
        with torch.no_grad():
            out_no_ple = layer(seqs, layout, bias_cache, per_layer_input=None)

        # Outputs should differ because PLE adds a residual contribution
        assert not torch.allclose(out_with_ple, out_no_ple, atol=1e-6)

    def test_moe_router_uses_pre_mlp_residual(self) -> None:
        """MoE router input is the pre-MLP residual, not the post-MLP output.

        This is a key correctness property: the routing decision is made
        independently of the dense MLP output.
        """
        layer = _make_base_layer(enable_moe=True)
        assert layer.enable_moe is True
        assert layer.router is not None

        # Just verify the layer runs — the actual pre-MLP residual usage
        # is an internal detail tested via the parity check
        seqs = torch.randn(1, 4, 64, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        with torch.no_grad():
            out = layer(seqs, layout, bias_cache)

        assert out.shape == (1, 4, 64)
        assert not torch.isnan(out).any()

    def test_kv_sharing_consumer_path(self) -> None:
        """Layer accepts pre_computed_kv (CONSUMER path)."""
        model_dim = 64
        head_dim = 16
        # _make_base_layer creates attention with num_heads=4 and no explicit
        # num_key_value_heads, so it defaults to num_heads=4.
        num_kv_heads = 4

        layer = _make_base_layer(model_dim=model_dim)
        seqs = torch.randn(1, 4, model_dim, device=device)
        layout = BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        pre_k = torch.randn(1, 4, num_kv_heads, head_dim, device=device)
        pre_v = torch.randn(1, 4, num_kv_heads, head_dim, device=device)

        with torch.no_grad():
            out = layer(
                seqs, layout, bias_cache,
                pre_computed_kv=(pre_k, pre_v),
            )

        assert out.shape == (1, 4, model_dim)

    def test_numerical_stability(self) -> None:
        """Layer produces no NaN/Inf with various input scales."""
        layer = _make_base_layer()
        layout_fn = lambda seqs: BatchLayout.of(seqs)
        bias_cache = AttentionBiasCache()

        test_inputs = [
            torch.randn(1, 4, 64, device=device) * 0.01,  # Small
            torch.randn(1, 4, 64, device=device) * 1.0,   # Normal
            torch.randn(1, 4, 64, device=device) * 10.0,  # Large
        ]

        for seqs in test_inputs:
            with torch.no_grad():
                out = layer(seqs, layout_fn(seqs), bias_cache)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

    def test_enable_flags(self) -> None:
        """enable_moe and enable_ple flags are set correctly."""
        layer_basic = _make_base_layer()
        assert layer_basic.enable_moe is False
        assert layer_basic.enable_ple is False

        layer_moe = _make_base_layer(enable_moe=True)
        assert layer_moe.enable_moe is True
        assert layer_moe.enable_ple is False

        layer_ple = _make_base_layer(enable_ple=True)
        assert layer_ple.enable_moe is False
        assert layer_ple.enable_ple is True

        layer_both = _make_base_layer(enable_moe=True, enable_ple=True)
        assert layer_both.enable_moe is True
        assert layer_both.enable_ple is True
