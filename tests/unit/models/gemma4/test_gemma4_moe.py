# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Gemma 4 MoE (router and experts)."""

from __future__ import annotations

import torch

from fairseq2.models.gemma4.moe import Gemma4Experts, Gemma4Router
from tests.common import assert_close, device


class TestGemma4Router:
    """Test Gemma4Router module."""

    def test_forward_output_shapes(self) -> None:
        """Router returns correct shapes: probs(T,E), weights(T,K), indices(T,K)."""
        router = Gemma4Router(
            model_dim=32, num_experts=8, top_k=2
        ).to(device)

        x = torch.randn(10, 32, device=device)

        with torch.no_grad():
            probs, weights, indices = router(x)

        assert probs.shape == (10, 8)  # (T, E)
        assert weights.shape == (10, 2)  # (T, K)
        assert indices.shape == (10, 2)  # (T, K)

    def test_router_probs_are_softmax(self) -> None:
        """Router probabilities should sum to 1 (softmax output)."""
        router = Gemma4Router(
            model_dim=32, num_experts=8, top_k=2
        ).to(device)

        x = torch.randn(10, 32, device=device)

        with torch.no_grad():
            probs, _, _ = router(x)

        sums = probs.sum(dim=-1)
        assert_close(sums, torch.ones(10, device=device), atol=1e-5)

    def test_top_k_weights_renormalized(self) -> None:
        """After renormalization, top-k weights sum to per_expert_scale-weighted values.

        Before per_expert_scale, the renormalized weights sum to 1.0.
        After per_expert_scale, the sum depends on the scale values.
        With default init (scale=ones), they should still sum close to 1.
        """
        router = Gemma4Router(
            model_dim=32, num_experts=8, top_k=2
        ).to(device)

        x = torch.randn(10, 32, device=device)

        with torch.no_grad():
            _, weights, indices = router(x)

        # With per_expert_scale initialized to ones, weighted sum ≈ 1.0
        # (exact equality depends on per_expert_scale at the selected indices)
        sums = weights.sum(dim=-1)
        assert_close(sums, torch.ones(10, device=device), atol=1e-5)

    def test_indices_within_range(self) -> None:
        """Top-k indices should be in [0, num_experts)."""
        num_experts = 8
        router = Gemma4Router(
            model_dim=32, num_experts=num_experts, top_k=2
        ).to(device)

        x = torch.randn(10, 32, device=device)

        with torch.no_grad():
            _, _, indices = router(x)

        assert indices.min() >= 0
        assert indices.max() < num_experts

    def test_scalar_root_size(self) -> None:
        """scalar_root_size should be model_dim^(-0.5)."""
        model_dim = 64
        router = Gemma4Router(
            model_dim=model_dim, num_experts=8, top_k=2
        )
        assert abs(router.scalar_root_size - model_dim ** -0.5) < 1e-10

    def test_norm_has_no_learnable_scale(self) -> None:
        """Router norm should be RMSNorm without learnable scale."""
        router = Gemma4Router(model_dim=32, num_experts=8, top_k=2)
        norm_params = list(router.norm.parameters())
        assert len(norm_params) == 0, "Router norm should have no learnable parameters"

    def test_per_expert_scale_effect(self) -> None:
        """per_expert_scale modulates the routing weights."""
        router = Gemma4Router(
            model_dim=32, num_experts=4, top_k=2
        ).to(device)

        x = torch.randn(10, 32, device=device)

        with torch.no_grad():
            _, weights_default, indices = router(x)

        # Double the per_expert_scale and re-run
        router.per_expert_scale.data.fill_(2.0)

        with torch.no_grad():
            _, weights_scaled, _ = router(x)

        # Weights should be approximately 2x the default (since per_expert_scale doubled)
        assert_close(weights_scaled, weights_default * 2.0, atol=1e-5)


class TestGemma4Experts:
    """Test Gemma4Experts module."""

    def test_forward_output_shape(self) -> None:
        """Experts output shape matches input shape (T, D)."""
        experts = Gemma4Experts(
            model_dim=32, num_experts=4, moe_intermediate_size=16
        ).to(device)
        torch.nn.init.normal_(experts.gate_up_proj, std=0.01)
        torch.nn.init.normal_(experts.down_proj, std=0.01)

        T = 6
        x = torch.randn(T, 32, device=device)
        indices = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [0, 3], [1, 0], [3, 2]], device=device
        )
        weights = torch.ones(T, 2, device=device) * 0.5

        with torch.no_grad():
            out = experts(x, indices, weights)

        assert out.shape == (T, 32)

    def test_weighted_output(self) -> None:
        """Output is weighted by routing weights — zero weight means no contribution."""
        experts = Gemma4Experts(
            model_dim=16, num_experts=4, moe_intermediate_size=8
        ).to(device)
        torch.nn.init.normal_(experts.gate_up_proj, std=0.01)
        torch.nn.init.normal_(experts.down_proj, std=0.01)

        T = 4
        x = torch.randn(T, 16, device=device)
        indices = torch.zeros(T, 2, dtype=torch.long, device=device)
        weights_nonzero = torch.ones(T, 2, device=device) * 0.5
        weights_zero = torch.zeros(T, 2, device=device)

        with torch.no_grad():
            out_nonzero = experts(x, indices, weights_nonzero)
            out_zero = experts(x, indices, weights_zero)

        assert_close(out_zero, torch.zeros_like(out_zero), atol=1e-6)
        assert out_nonzero.abs().mean() > 1e-6

    def test_gate_up_proj_shape(self) -> None:
        """gate_up_proj should have shape (E, 2*I, D)."""
        num_experts = 4
        model_dim = 32
        moe_inner = 16
        experts = Gemma4Experts(
            model_dim=model_dim,
            num_experts=num_experts,
            moe_intermediate_size=moe_inner,
        )
        assert experts.gate_up_proj.shape == (num_experts, 2 * moe_inner, model_dim)

    def test_down_proj_shape(self) -> None:
        """down_proj should have shape (E, D, I)."""
        num_experts = 4
        model_dim = 32
        moe_inner = 16
        experts = Gemma4Experts(
            model_dim=model_dim,
            num_experts=num_experts,
            moe_intermediate_size=moe_inner,
        )
        assert experts.down_proj.shape == (num_experts, model_dim, moe_inner)

    def test_gelu_tanh_activation(self) -> None:
        """Experts use gelu_pytorch_tanh by default."""
        experts = Gemma4Experts(
            model_dim=16, num_experts=2, moe_intermediate_size=8
        )
        # Test the activation function directly
        x = torch.randn(4, 8)
        expected = torch.nn.functional.gelu(x, approximate="tanh")
        actual = experts._act_fn(x)
        assert_close(actual, expected, atol=1e-6)

    def test_large_expert_count(self) -> None:
        """Test with a larger number of experts (like E4B's 128)."""
        num_experts = 16
        top_k = 4
        experts = Gemma4Experts(
            model_dim=32, num_experts=num_experts, moe_intermediate_size=8
        ).to(device)
        torch.nn.init.normal_(experts.gate_up_proj, std=0.01)
        torch.nn.init.normal_(experts.down_proj, std=0.01)

        T = 10
        x = torch.randn(T, 32, device=device)
        indices = torch.randint(0, num_experts, (T, top_k), device=device)
        weights = torch.ones(T, top_k, device=device) / top_k

        with torch.no_grad():
            out = experts(x, indices, weights)

        assert out.shape == (T, 32)
        assert not torch.isnan(out).any()
