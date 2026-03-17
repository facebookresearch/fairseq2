# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from fairseq2.models.gemma3n.decoder_layer import Gemma3nLAuReL
from fairseq2.nn import RMSNorm
from tests.common import device


class TestGemma3nLAuReL:
    """Test Learned Augmented Residual Layer (LAuReL)."""

    def test_low_rank_bottleneck(self) -> None:
        """Verify LAuReL creates low-rank bottleneck: D → rank → D."""
        model_dim, rank = 128, 16

        layer_norm = RMSNorm(model_dim, bias=False, device=device)
        laurel = Gemma3nLAuReL(
            model_dim=model_dim,
            rank=rank,
            layer_norm=layer_norm,
            device=device,
        )

        # Check projection dimensions
        assert laurel.linear_left.weight.shape == (rank, model_dim)
        assert laurel.linear_right.weight.shape == (model_dim, rank)

    def test_forward_preserves_shape(self) -> None:
        """Verify LAuReL forward preserves input shape."""
        model_dim, rank = 64, 16
        batch_size, seq_len = 2, 8

        layer_norm = RMSNorm(model_dim, bias=False, device=device)
        laurel = Gemma3nLAuReL(
            model_dim=model_dim,
            rank=rank,
            layer_norm=layer_norm,
            device=device,
        )

        hidden_states = torch.randn(batch_size, seq_len, model_dim, device=device)

        with torch.no_grad():
            output = laurel(hidden_states)

        assert output.shape == (batch_size, seq_len, model_dim)
        assert output.dtype == hidden_states.dtype

    def test_residual_connection(self) -> None:
        """Verify LAuReL includes residual connection."""
        model_dim, rank = 64, 16
        batch_size, seq_len = 2, 8

        layer_norm = RMSNorm(model_dim, bias=False, device=device)
        laurel = Gemma3nLAuReL(
            model_dim=model_dim,
            rank=rank,
            layer_norm=layer_norm,
            device=device,
        )

        hidden_states = torch.randn(batch_size, seq_len, model_dim, device=device)

        with torch.no_grad():
            output = laurel(hidden_states)

        # Output should differ from input (learned transformation applied)
        assert not torch.allclose(output, hidden_states, rtol=1e-3)

        # But should contain contribution from input (residual)
        # Set weights to zero to verify residual path exists
        laurel.linear_left.weight.data.zero_()  # type: ignore[operator]
        laurel.linear_right.weight.data.zero_()  # type: ignore[operator]

        with torch.no_grad():
            output_zero_weights = laurel(hidden_states)

        # With zero weights, output should equal input (pure residual)
        assert torch.allclose(output_zero_weights, hidden_states, rtol=1e-5)

    def test_layer_norm_applied(self) -> None:
        """Verify layer normalization applied to LAuReL output."""
        model_dim, rank = 64, 16

        layer_norm = RMSNorm(model_dim, bias=False, device=device)
        laurel = Gemma3nLAuReL(
            model_dim=model_dim,
            rank=rank,
            layer_norm=layer_norm,
            device=device,
        )

        # Verify layer norm is used
        assert laurel.post_laurel_norm is layer_norm

    def test_rank_dimensionality(self) -> None:
        """Verify various rank values work correctly."""
        model_dim = 128
        test_ranks = [8, 16, 32, 64]

        for rank in test_ranks:
            layer_norm = RMSNorm(model_dim, bias=False, device=device)
            laurel = Gemma3nLAuReL(
                model_dim=model_dim,
                rank=rank,
                layer_norm=layer_norm,
                device=device,
            )

            batch_size, seq_len = 1, 4
            hidden_states = torch.randn(batch_size, seq_len, model_dim, device=device)

            with torch.no_grad():
                output = laurel(hidden_states)

            assert output.shape == (batch_size, seq_len, model_dim)

    def test_typical_gemma3n_config(self) -> None:
        """Verify LAuReL with typical Gemma3n config (rank=64, model_dim=2048)."""
        model_dim, rank = 2048, 64
        batch_size, seq_len = 1, 4

        layer_norm = RMSNorm(model_dim, bias=False, device=device)
        laurel = Gemma3nLAuReL(
            model_dim=model_dim,
            rank=rank,
            layer_norm=layer_norm,
            device=device,
        )

        hidden_states = torch.randn(batch_size, seq_len, model_dim, device=device)

        with torch.no_grad():
            output = laurel(hidden_states)

        assert output.shape == (batch_size, seq_len, model_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_no_bias_in_projections(self) -> None:
        """Verify LAuReL projections have no bias."""
        model_dim, rank = 64, 16

        layer_norm = RMSNorm(model_dim, bias=False, device=device)
        laurel = Gemma3nLAuReL(
            model_dim=model_dim,
            rank=rank,
            layer_norm=layer_norm,
            device=device,
        )

        # Both projections should have no bias
        assert laurel.linear_left.bias is None
        assert laurel.linear_right.bias is None

    def test_forward_numerical_stability(self) -> None:
        """Verify LAuReL forward is numerically stable."""
        model_dim, rank = 128, 32
        batch_size, seq_len = 2, 16

        layer_norm = RMSNorm(model_dim, bias=False, device=device)
        laurel = Gemma3nLAuReL(
            model_dim=model_dim,
            rank=rank,
            layer_norm=layer_norm,
            device=device,
        )

        # Test with various input ranges
        test_inputs = [
            torch.randn(batch_size, seq_len, model_dim, device=device) * 0.01,  # Small
            torch.randn(batch_size, seq_len, model_dim, device=device) * 1.0,  # Normal
            torch.randn(batch_size, seq_len, model_dim, device=device) * 10.0,  # Large
        ]

        for hidden_states in test_inputs:
            with torch.no_grad():
                output = laurel(hidden_states)

            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            assert output.shape == hidden_states.shape

    def test_batch_independence(self) -> None:
        """Verify LAuReL processes batch items independently."""
        model_dim, rank = 64, 16
        batch_size, seq_len = 4, 8

        layer_norm = RMSNorm(model_dim, bias=False, device=device)
        laurel = Gemma3nLAuReL(
            model_dim=model_dim,
            rank=rank,
            layer_norm=layer_norm,
            device=device,
        )

        # Process full batch
        hidden_states = torch.randn(batch_size, seq_len, model_dim, device=device)

        with torch.no_grad():
            batch_output = laurel(hidden_states)

        # Process items individually
        with torch.no_grad():
            individual_outputs = [
                laurel(hidden_states[i : i + 1]) for i in range(batch_size)
            ]

        # Results should match
        for i in range(batch_size):
            assert torch.allclose(
                batch_output[i : i + 1], individual_outputs[i], rtol=1e-5
            )
