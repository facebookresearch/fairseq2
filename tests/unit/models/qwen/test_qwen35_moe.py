# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from fairseq2.models.qwen.moe import Qwen35Experts, Qwen35MoeBlock, Qwen35TopKRouter
from tests.common import assert_close, device


class TestQwen35TopKRouter:
    def test_forward_output_shapes(self) -> None:
        """Router returns correct shapes: logits(T,E), weights(T,K), indices(T,K)."""
        router = Qwen35TopKRouter(num_experts=8, top_k=2, model_dim=32).to(device)

        x = torch.randn(10, 32, device=device)

        with torch.no_grad():
            logits, weights, indices = router(x)

        assert logits.shape == (10, 8)  # (T, E)
        assert weights.shape == (10, 2)  # (T, K)
        assert indices.shape == (10, 2)  # (T, K)

    def test_weights_sum_to_one(self) -> None:
        """Renormalized top-k weights sum to 1 per token."""
        router = Qwen35TopKRouter(num_experts=8, top_k=2, model_dim=32).to(device)

        x = torch.randn(10, 32, device=device)

        with torch.no_grad():
            _, weights, _ = router(x)

        sums = weights.sum(dim=-1)
        assert_close(sums, torch.ones(10, device=device), atol=1e-5)

    def test_logits_are_softmax(self) -> None:
        """Router logits are valid probability distribution (sum to 1, non-negative)."""
        router = Qwen35TopKRouter(num_experts=8, top_k=2, model_dim=32).to(device)

        x = torch.randn(10, 32, device=device)

        with torch.no_grad():
            logits, _, _ = router(x)

        assert (logits >= 0).all()

        sums = logits.sum(dim=-1)
        assert_close(sums, torch.ones(10, device=device), atol=1e-5)


class TestQwen35Experts:
    def test_forward_output_shape(self) -> None:
        """Experts output shape matches input shape (T, D)."""
        experts = Qwen35Experts(
            num_experts=4, model_dim=32, expert_inner_dim=16
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
        experts = Qwen35Experts(
            num_experts=4, model_dim=16, expert_inner_dim=8
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


class TestQwen35MoeBlock:
    def test_forward_output_shape(self) -> None:
        """MoeBlock output shape matches input (B, S, D)."""
        moe = Qwen35MoeBlock(
            model_dim=32,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
        ).to(device)

        seqs = torch.randn(2, 8, 32, device=device)

        with torch.no_grad():
            out = moe(seqs)

        assert out.shape == (2, 8, 32)

    def test_shared_expert_contributes(self) -> None:
        """Shared expert output is non-zero (sigmoid gate blending)."""
        moe = Qwen35MoeBlock(
            model_dim=32,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
        ).to(device)

        seqs = torch.randn(1, 4, 32, device=device)

        with torch.no_grad():
            out = moe(seqs)

        assert out.abs().mean() > 1e-6

    def test_drop_in_ffn_replacement(self) -> None:
        """MoeBlock inherits FeedForwardNetwork and can be used as drop-in."""
        from fairseq2.models.transformer import FeedForwardNetwork

        moe = Qwen35MoeBlock(
            model_dim=32,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
        )

        assert isinstance(moe, FeedForwardNetwork)
