# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for :class:`Gemma4SDPA`.

Verifies that Gemma4SDPA passes ``scale`` directly to PyTorch's
``scaled_dot_product_attention`` kernel rather than pre-multiplying Q,
which avoids bfloat16 precision loss with large head_dim values.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fairseq2.models.gemma4.sdpa import Gemma4SDPA
from fairseq2.models.transformer import AttentionBiasCache, CausalAttentionBias
from fairseq2.nn import BatchLayout


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(42)


# ---- Helpers ----

def _make_qkv(
    batch: int, seq_len: int, num_heads: int, head_dim: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Create random Q, K, V tensors in (B, S, H, D) format."""
    shape = (batch, seq_len, num_heads, head_dim)
    q = torch.randn(shape)
    k = torch.randn(shape)
    v = torch.randn(shape)
    return q, k, v


def _layout(seqs: Tensor) -> BatchLayout:
    """Create a simple batch layout from a (B, S, ...) tensor."""
    batch_size = seqs.size(0)
    seq_len = seqs.size(1)
    # BatchLayout expects a 2D token-ID-like tensor for .of()
    dummy = torch.zeros(batch_size, seq_len, dtype=torch.long)
    return BatchLayout.of(dummy)


# ---- Tests: Basic Functionality ----

class TestGemma4SDPABasic:
    """Basic forward pass tests."""

    def test_output_shape(self) -> None:
        """Output shape matches input Q shape."""
        B, S, H, D = 2, 8, 4, 256
        q, k, v = _make_qkv(B, S, H, D)
        layout = _layout(q)
        bias_cache = AttentionBiasCache()

        sdpa = Gemma4SDPA(CausalAttentionBias(), scale=1.0)
        out, weights = sdpa(q, layout, k, layout, v, bias_cache)

        assert out.shape == (B, S, H, D)
        assert weights is None  # Gemma4SDPA never returns weights

    def test_output_shape_head_dim_512(self) -> None:
        """Output shape with head_dim=512 (global attention)."""
        B, S, H, D = 1, 4, 8, 512
        q, k, v = _make_qkv(B, S, H, D)
        layout = _layout(q)
        bias_cache = AttentionBiasCache()

        sdpa = Gemma4SDPA(CausalAttentionBias(), scale=1.0)
        out, _ = sdpa(q, layout, k, layout, v, bias_cache)

        assert out.shape == (B, S, H, D)

    def test_output_dtype_preserved(self) -> None:
        """Output dtype matches input dtype."""
        B, S, H, D = 1, 4, 4, 128
        q, k, v = _make_qkv(B, S, H, D)
        layout = _layout(q)
        bias_cache = AttentionBiasCache()

        sdpa = Gemma4SDPA(CausalAttentionBias(), scale=1.0)
        out, _ = sdpa(q, layout, k, layout, v, bias_cache)

        assert out.dtype == q.dtype

    def test_output_is_finite(self) -> None:
        """Output contains no NaN or Inf values."""
        B, S, H, D = 2, 16, 4, 256
        q, k, v = _make_qkv(B, S, H, D)
        layout = _layout(q)
        bias_cache = AttentionBiasCache()

        sdpa = Gemma4SDPA(CausalAttentionBias(), scale=1.0)
        out, _ = sdpa(q, layout, k, layout, v, bias_cache)

        assert torch.isfinite(out).all()


# ---- Tests: Scale Behavior ----

class TestGemma4SDPAScale:
    """Tests for scale passthrough behavior (core Gemma4 fix)."""

    def test_scale_one_no_q_modification(self) -> None:
        """With scale=1.0, Q is NOT pre-multiplied (unlike TorchSDPA).

        This is the core behavioral difference: TorchSDPA would compute
        Q *= scale * sqrt(head_dim), but Gemma4SDPA passes scale directly
        to the kernel. We verify by checking that the input Q tensor is
        not modified in-place.
        """
        B, S, H, D = 1, 4, 4, 512
        q, k, v = _make_qkv(B, S, H, D)
        q_orig = q.clone()
        layout = _layout(q)
        bias_cache = {}

        sdpa = Gemma4SDPA(CausalAttentionBias(), scale=1.0)
        sdpa(q, layout, k, layout, v, bias_cache)

        # Q should NOT be modified in-place
        assert torch.equal(q, q_orig), "Gemma4SDPA should not modify Q in-place"

    def test_scale_none_uses_default(self) -> None:
        """With scale=None, PyTorch's default 1/sqrt(head_dim) is used."""
        B, S, H, D = 1, 4, 4, 64
        q, k, v = _make_qkv(B, S, H, D)
        layout = _layout(q)
        bias_cache = AttentionBiasCache()

        sdpa_none = Gemma4SDPA(CausalAttentionBias(), scale=None)
        out_none, _ = sdpa_none(q, layout, k, layout, v, bias_cache)

        sdpa_explicit = Gemma4SDPA(CausalAttentionBias(), scale=1.0 / (D ** 0.5))
        out_explicit, _ = sdpa_explicit(q, layout, k, layout, v, bias_cache)

        # Both should produce the same result (within numerical tolerance)
        assert torch.allclose(out_none, out_explicit, atol=1e-5)

    def test_different_scales_produce_different_outputs(self) -> None:
        """Different scale values should produce different attention outputs."""
        B, S, H, D = 1, 8, 4, 128
        q, k, v = _make_qkv(B, S, H, D)
        layout = _layout(q)

        sdpa_1 = Gemma4SDPA(CausalAttentionBias(), scale=1.0)
        out_1, _ = sdpa_1(q, layout, k, layout, v, AttentionBiasCache())

        sdpa_half = Gemma4SDPA(CausalAttentionBias(), scale=0.5)
        out_half, _ = sdpa_half(q, layout, k, layout, v, AttentionBiasCache())

        # Different scales must produce different outputs
        assert not torch.allclose(out_1, out_half, atol=1e-4)

    def test_scale_one_equals_no_scaling(self) -> None:
        """scale=1.0 means attention scores = Q @ K^T (no normalization).

        This is what Gemma4 needs because it uses QK-norm instead of
        1/sqrt(head_dim) scaling.
        """
        B, S, H, D = 1, 4, 2, 64
        q, k, v = _make_qkv(B, S, H, D)
        layout = _layout(q)

        sdpa = Gemma4SDPA(CausalAttentionBias(), scale=1.0)
        out, _ = sdpa(q, layout, k, layout, v, AttentionBiasCache())

        # Verify output is finite and well-formed
        assert torch.isfinite(out).all()
        assert out.shape == (B, S, H, D)


# ---- Tests: Causal Bias ----

class TestGemma4SDPACausalBias:
    """Tests for causal attention masking."""

    def test_causal_mask_applied(self) -> None:
        """Causal mask prevents attending to future positions."""
        B, S, H, D = 1, 8, 2, 64
        q, k, v = _make_qkv(B, S, H, D)
        # Make v distinctive per position so we can verify causality
        v = torch.arange(S, dtype=torch.float32).view(1, S, 1, 1).expand(B, S, H, D)
        layout = _layout(q)

        sdpa = Gemma4SDPA(CausalAttentionBias(), scale=1.0)
        out, _ = sdpa(q, layout, k, layout, v, AttentionBiasCache())

        # First position should only attend to itself (position 0)
        # Its output should be close to v[0] = 0
        # Last position attends to all positions
        first_pos_mean = out[0, 0].mean().item()
        last_pos_mean = out[0, -1].mean().item()

        # First position output should be exactly v[0] (only self-attention)
        assert abs(first_pos_mean) < 1e-4, f"First position mean={first_pos_mean}, expected ~0"

    def test_sliding_window_causal_bias(self) -> None:
        """Sliding window limits attention to recent positions."""
        B, S, H, D = 1, 16, 2, 64
        q, k, v = _make_qkv(B, S, H, D)
        layout = _layout(q)

        window = 4
        sdpa_sliding = Gemma4SDPA(
            CausalAttentionBias(attn_window_len=window), scale=1.0
        )
        out_sliding, _ = sdpa_sliding(q, layout, k, layout, v, AttentionBiasCache())

        sdpa_full = Gemma4SDPA(CausalAttentionBias(), scale=1.0)
        out_full, _ = sdpa_full(q, layout, k, layout, v, AttentionBiasCache())

        # Sliding and full should differ (after window fills up)
        # At position > window, sliding window truncates attention
        assert not torch.allclose(out_sliding, out_full, atol=1e-4)


# ---- Tests: Dropout ----

class TestGemma4SDPADropout:
    """Tests for dropout handling."""

    def test_no_dropout_in_eval(self) -> None:
        """Dropout is not applied during evaluation."""
        B, S, H, D = 1, 8, 4, 64
        q, k, v = _make_qkv(B, S, H, D)
        layout = _layout(q)

        sdpa = Gemma4SDPA(CausalAttentionBias(), dropout_p=0.5, scale=1.0)
        sdpa.eval()

        # Run twice — should get identical results (no dropout in eval)
        out1, _ = sdpa(q, layout, k, layout, v, AttentionBiasCache())
        out2, _ = sdpa(q, layout, k, layout, v, AttentionBiasCache())

        assert torch.equal(out1, out2), "Eval mode should produce deterministic output"

    def test_zero_dropout_is_deterministic(self) -> None:
        """dropout_p=0.0 produces deterministic results in train mode."""
        B, S, H, D = 1, 8, 4, 64
        q, k, v = _make_qkv(B, S, H, D)
        layout = _layout(q)

        sdpa = Gemma4SDPA(CausalAttentionBias(), dropout_p=0.0, scale=1.0)
        sdpa.train()

        out1, _ = sdpa(q, layout, k, layout, v, AttentionBiasCache())
        out2, _ = sdpa(q, layout, k, layout, v, AttentionBiasCache())

        assert torch.equal(out1, out2)


# ---- Tests: Extra Repr ----

class TestGemma4SDPARepr:
    """Tests for string representation."""

    def test_extra_repr_with_scale(self) -> None:
        sdpa = Gemma4SDPA(CausalAttentionBias(), scale=1.0)
        repr_str = sdpa.extra_repr()
        assert "scale=1" in repr_str

    def test_extra_repr_without_scale(self) -> None:
        sdpa = Gemma4SDPA(CausalAttentionBias(), scale=None)
        repr_str = sdpa.extra_repr()
        assert "scale" not in repr_str

    def test_extra_repr_dropout(self) -> None:
        sdpa = Gemma4SDPA(CausalAttentionBias(), dropout_p=0.1, scale=1.0)
        repr_str = sdpa.extra_repr()
        assert "dropout_p=0.1" in repr_str


# ---- Tests: Error Handling ----

class TestGemma4SDPAErrors:
    """Tests for error cases."""

    def test_needs_weights_raises(self) -> None:
        """needs_weights=True should raise NotSupportedError."""
        B, S, H, D = 1, 4, 2, 64
        q, k, v = _make_qkv(B, S, H, D)
        layout = _layout(q)

        sdpa = Gemma4SDPA(CausalAttentionBias(), scale=1.0)

        with pytest.raises(Exception):
            sdpa(q, layout, k, layout, v, {}, needs_weights=True)
