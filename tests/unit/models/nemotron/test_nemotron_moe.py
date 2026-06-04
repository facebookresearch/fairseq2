# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for NemotronH MoE (Mixture of Experts) module."""

from __future__ import annotations

import pytest
import torch

from fairseq2.models.nemotron.moe import (
    NemotronHExpert,
    NemotronHMoE,
    NemotronHTopKRouter,
    SquaredReLU,
)


class TestSquaredReLU:
    def test_positive_input(self) -> None:
        act = SquaredReLU()
        x = torch.tensor([2.0, 3.0, 4.0])
        out = act(x)
        expected = torch.tensor([4.0, 9.0, 16.0])
        assert torch.allclose(out, expected)

    def test_negative_input(self) -> None:
        act = SquaredReLU()
        x = torch.tensor([-1.0, -2.0, 0.0])
        out = act(x)
        expected = torch.zeros(3)
        assert torch.allclose(out, expected)

    def test_mixed_input(self) -> None:
        act = SquaredReLU()
        x = torch.tensor([-1.0, 2.0, 0.0, 3.0])
        out = act(x)
        expected = torch.tensor([0.0, 4.0, 0.0, 9.0])
        assert torch.allclose(out, expected)


class TestNemotronHExpert:
    def test_output_shape(self) -> None:
        expert = NemotronHExpert(128, 64)
        x = torch.randn(4, 128)
        out = expert(x)
        assert out.shape == (4, 128)

    def test_no_bias(self) -> None:
        expert = NemotronHExpert(128, 64, bias=False)
        assert expert.up_proj.bias is None
        assert expert.down_proj.bias is None

    def test_with_bias(self) -> None:
        expert = NemotronHExpert(128, 64, bias=True)
        assert expert.up_proj.bias is not None
        assert expert.down_proj.bias is not None


class TestNemotronHTopKRouter:
    def test_output_shapes(self) -> None:
        router = NemotronHTopKRouter(128, num_experts=16, top_k=4)
        x = torch.randn(8, 128)
        weights, indices = router(x)
        assert weights.shape == (8, 4)
        assert indices.shape == (8, 4)

    def test_weights_positive(self) -> None:
        router = NemotronHTopKRouter(128, num_experts=16, top_k=4)
        x = torch.randn(8, 128)
        weights, _ = router(x)
        assert (weights >= 0).all()

    def test_scaling_factor(self) -> None:
        """Verify the routing weights are scaled by routed_scaling_factor."""
        router = NemotronHTopKRouter(
            128, num_experts=16, top_k=4, routed_scaling_factor=2.5
        )
        x = torch.randn(8, 128)
        weights, _ = router(x)
        # After normalization (sum=1) and scaling by 2.5,
        # sum of weights should be approximately 2.5
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.full_like(weight_sums, 2.5), atol=0.1)

    def test_bias_correction_buffer(self) -> None:
        router = NemotronHTopKRouter(128, num_experts=16, top_k=4)
        assert hasattr(router, "e_score_correction_bias")
        assert router.e_score_correction_bias.shape == (16,)
        assert router.e_score_correction_bias.dtype == torch.float32

    def test_different_topk(self) -> None:
        router = NemotronHTopKRouter(128, num_experts=32, top_k=6)
        x = torch.randn(4, 128)
        weights, indices = router(x)
        assert weights.shape == (4, 6)
        assert indices.shape == (4, 6)

    def test_expert_indices_valid(self) -> None:
        num_experts = 16
        router = NemotronHTopKRouter(128, num_experts=num_experts, top_k=4)
        x = torch.randn(8, 128)
        _, indices = router(x)
        assert (indices >= 0).all()
        assert (indices < num_experts).all()

    def test_expert_indices_unique_per_token(self) -> None:
        """Each token should select different experts."""
        router = NemotronHTopKRouter(128, num_experts=16, top_k=4)
        x = torch.randn(4, 128)
        _, indices = router(x)
        for i in range(4):
            assert len(set(indices[i].tolist())) == 4


class TestNemotronHMoE:
    @pytest.fixture
    def small_moe(self) -> NemotronHMoE:
        return NemotronHMoE(
            model_dim=128,
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=128,
        )

    def test_output_shape_2d(self, small_moe: NemotronHMoE) -> None:
        x = torch.randn(4, 128)
        out = small_moe(x)
        assert out.shape == (4, 128)

    def test_output_shape_3d(self, small_moe: NemotronHMoE) -> None:
        x = torch.randn(2, 8, 128)
        out = small_moe(x)
        assert out.shape == (2, 8, 128)

    def test_gradient_flow(self, small_moe: NemotronHMoE) -> None:
        x = torch.randn(2, 4, 128, requires_grad=True)
        out = small_moe(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_expert_count(self, small_moe: NemotronHMoE) -> None:
        assert len(small_moe.experts) == 8
        assert small_moe.shared_experts is not None

    def test_shared_expert_always_contributes(self) -> None:
        """Shared expert should always contribute to output."""
        moe = NemotronHMoE(
            model_dim=128,
            num_experts=4,
            num_experts_per_tok=1,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=64,
        )
        x = torch.randn(2, 4, 128)
        with torch.no_grad():
            out = moe(x)
        # Output should not be zero (shared expert always active)
        assert out.abs().sum() > 0

    def test_default_moe_config_params(self) -> None:
        """Test MoE with default NemotronH config dimensions."""
        moe = NemotronHMoE(
            model_dim=2688,
            num_experts=128,
            num_experts_per_tok=6,
            moe_intermediate_size=1856,
            shared_expert_intermediate_size=3712,
            routed_scaling_factor=2.5,
        )
        # Just verify it can be constructed
        assert moe.num_experts == 128
        assert moe.num_experts_per_tok == 6

        # Count parameters
        total = sum(p.numel() for p in moe.parameters())
        # Each expert: up(1856*2688) + down(2688*1856) = 2 * 1856 * 2688
        # 128 experts + 1 shared (3712*2688*2) + router (128*2688)
        print(f"MoE total params: {total:,}")
        assert total > 0
