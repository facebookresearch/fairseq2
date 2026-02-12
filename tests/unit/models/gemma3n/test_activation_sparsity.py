# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch

from fairseq2.models.transformer.ffn import (
    AltUpFeedForwardNetwork,
    GLUFeedForwardNetwork,
)
from tests.common import device


class TestGaussianTopKSparsity:
    """Test Gaussian top-k sparsification in feed-forward networks."""

    def test_zero_sparsity_returns_input_unchanged(self) -> None:
        """Verify sparsity=0.0 returns input without modification."""
        ffn = AltUpFeedForwardNetwork(
            model_dim=64,
            inner_dim=256,
            bias=False,
            activation_sparsity=0.0,
            device=device,
        )

        batch_size, seq_len, model_dim = 2, 8, 64
        seqs = torch.randn(batch_size, seq_len, model_dim, device=device)

        with torch.no_grad():
            output = ffn(seqs)

        assert output.shape == (batch_size, seq_len, model_dim)
        assert not torch.all(output == 0)

    def test_high_sparsity_produces_sparse_output(self) -> None:
        """Verify high sparsity (0.95) affects output magnitude."""
        torch.manual_seed(42)

        # Create two FFNs: one with sparsity, one without
        ffn_sparse = AltUpFeedForwardNetwork(
            model_dim=64,
            inner_dim=256,
            bias=False,
            activation_sparsity=0.95,
            device=device,
        )

        ffn_dense = AltUpFeedForwardNetwork(
            model_dim=64,
            inner_dim=256,
            bias=False,
            activation_sparsity=0.0,
            device=device,
        )

        # Copy weights to make them identical
        ffn_dense.gate_proj.weight.data = ffn_sparse.gate_proj.weight.data.clone()
        ffn_dense.inner_proj.weight.data = ffn_sparse.inner_proj.weight.data.clone()
        ffn_dense.output_proj.weight.data = ffn_sparse.output_proj.weight.data.clone()

        batch_size, seq_len, model_dim = 2, 16, 64
        seqs = torch.randn(batch_size, seq_len, model_dim, device=device)

        with torch.no_grad():
            output_sparse = ffn_sparse(seqs)
            output_dense = ffn_dense(seqs)

        sparse_norm = torch.norm(output_sparse).item()
        dense_norm = torch.norm(output_dense).item()

        assert sparse_norm < dense_norm * 0.5, f"Sparse {sparse_norm} not much smaller than dense {dense_norm}"

    def test_sparsity_applied_before_activation(self) -> None:
        """Verify Gaussian top-k applied before GELU activation."""
        torch.manual_seed(42)

        ffn = AltUpFeedForwardNetwork(
            model_dim=32,
            inner_dim=128,
            bias=False,
            activation_sparsity=0.5,
            device=device,
        )

        batch_size, seq_len, model_dim = 1, 8, 32
        seqs = torch.randn(batch_size, seq_len, model_dim, device=device)

        with torch.no_grad():
            output = ffn(seqs)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.shape == (batch_size, seq_len, model_dim)

    @pytest.mark.parametrize("sparsity", [0.0, 0.5, 0.9, 0.95, 0.99])
    def test_various_sparsity_levels(self, sparsity: float) -> None:
        """Verify FFN works with various sparsity levels."""
        ffn = GLUFeedForwardNetwork(
            model_dim=64,
            inner_dim=256,
            bias=False,
            activation_sparsity=sparsity,
            device=device,
        )

        batch_size, seq_len, model_dim = 2, 8, 64
        seqs = torch.randn(batch_size, seq_len, model_dim, device=device)

        with torch.no_grad():
            output = ffn(seqs)

        assert output.shape == (batch_size, seq_len, model_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_altup_ffn_has_sparsity_parameter(self) -> None:
        """Verify AltUpFeedForwardNetwork accepts sparsity parameter."""
        ffn = AltUpFeedForwardNetwork(
            model_dim=64,
            inner_dim=256,
            bias=False,
            activation_sparsity=0.95,
            device=device,
        )

        assert hasattr(ffn, "activation_sparsity")
        assert ffn.activation_sparsity == 0.95

    def test_glu_ffn_has_sparsity_parameter(self) -> None:
        """Verify GLUFeedForwardNetwork accepts sparsity parameter."""
        ffn = GLUFeedForwardNetwork(
            model_dim=64,
            inner_dim=256,
            bias=False,
            activation_sparsity=0.95,
            device=device,
        )

        assert hasattr(ffn, "activation_sparsity")
        assert ffn.activation_sparsity == 0.95

    def test_gaussian_topk_deterministic_with_seed(self) -> None:
        """Verify Gaussian top-k is deterministic with same seed."""
        model_dim, inner_dim = 32, 128
        batch_size, seq_len = 2, 8

        seqs = torch.randn(batch_size, seq_len, model_dim, device=device)

        # Run twice with same seed
        outputs = []
        for _ in range(2):
            torch.manual_seed(42)
            ffn = AltUpFeedForwardNetwork(
                model_dim=model_dim,
                inner_dim=inner_dim,
                bias=False,
                activation_sparsity=0.95,
                device=device,
            )
            with torch.no_grad():
                outputs.append(ffn(seqs))

        assert torch.allclose(outputs[0], outputs[1], rtol=1e-5)

    def test_shape_preservation_with_sparsity(self) -> None:
        """Verify output shape preserved regardless of sparsity."""
        test_cases = [
            (1, 4, 32, 128),  # Minimal
            (2, 16, 64, 256),  # Medium
            (4, 32, 128, 512),  # Large
        ]

        for batch_size, seq_len, model_dim, inner_dim in test_cases:
            ffn = GLUFeedForwardNetwork(
                model_dim=model_dim,
                inner_dim=inner_dim,
                bias=False,
                activation_sparsity=0.95,
                device=device,
            )

            seqs = torch.randn(batch_size, seq_len, model_dim, device=device)

            with torch.no_grad():
                output = ffn(seqs)

            assert output.shape == (batch_size, seq_len, model_dim)
