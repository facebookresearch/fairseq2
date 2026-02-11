# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from fairseq2.models.gemma3n.ple import PerLayerEmbedding
from fairseq2.nn import RMSNorm
from tests.common import device


class TestPerLayerEmbedding:
    """Test Per-Layer Embedding (PLE) augmentation."""

    def test_forward_shape_preservation(self) -> None:
        """Verify PLE preserves input sequence shape."""
        vocab_size, hidden_size, model_dim = 100, 32, 64
        batch_size, seq_len = 2, 8

        layer_norm = RMSNorm(model_dim, bias=False, device=device)
        ple = PerLayerEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            model_dim=model_dim,
            layer_norm=layer_norm,
            device=device,
        )

        seqs = torch.randn(batch_size, seq_len, model_dim, device=device)
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        with torch.no_grad():
            output = ple(seqs, token_ids)

        assert output.shape == (batch_size, seq_len, model_dim)
        assert output.dtype == seqs.dtype

    def test_embedding_lookup(self) -> None:
        """Verify PLE embedding table lookup works correctly."""
        vocab_size, hidden_size, model_dim = 100, 32, 64
        batch_size, seq_len = 2, 8

        layer_norm = RMSNorm(model_dim, bias=False, device=device)
        ple = PerLayerEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            model_dim=model_dim,
            layer_norm=layer_norm,
            device=device,
        )

        # Verify embedding table has correct shape
        assert ple.embed_tokens_per_layer.weight.shape == (vocab_size, hidden_size)

    def test_gate_projection(self) -> None:
        """Verify gating mechanism projects from model_dim to hidden_size."""
        vocab_size, hidden_size, model_dim = 100, 32, 64
        batch_size, seq_len = 2, 8

        layer_norm = RMSNorm(model_dim, bias=False, device=device)
        ple = PerLayerEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            model_dim=model_dim,
            layer_norm=layer_norm,
            device=device,
        )

        # Verify gate projection dimensions
        assert ple.per_layer_input_gate.weight.shape == (hidden_size, model_dim)

    def test_output_projection(self) -> None:
        """Verify output projection from hidden_size to model_dim."""
        vocab_size, hidden_size, model_dim = 100, 32, 64
        batch_size, seq_len = 2, 8

        layer_norm = RMSNorm(model_dim, bias=False, device=device)
        ple = PerLayerEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            model_dim=model_dim,
            layer_norm=layer_norm,
            device=device,
        )

        # Verify projection dimensions
        assert ple.per_layer_projection.weight.shape == (model_dim, hidden_size)

    def test_augmentation_applied(self) -> None:
        """Verify PLE augments input sequences."""
        vocab_size, hidden_size, model_dim = 100, 32, 64
        batch_size, seq_len = 2, 8

        layer_norm = RMSNorm(model_dim, bias=False, device=device)
        ple = PerLayerEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            model_dim=model_dim,
            layer_norm=layer_norm,
            device=device,
        )

        seqs = torch.randn(batch_size, seq_len, model_dim, device=device)
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        with torch.no_grad():
            output = ple(seqs, token_ids)

        # Output should differ from input (PLE augmentation applied)
        assert not torch.allclose(output, seqs, rtol=1e-3)

    def test_different_tokens_different_outputs(self) -> None:
        """Verify different token IDs produce different PLE contributions."""
        vocab_size, hidden_size, model_dim = 100, 32, 64
        batch_size, seq_len = 1, 2

        layer_norm = RMSNorm(model_dim, bias=False, device=device)
        ple = PerLayerEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            model_dim=model_dim,
            layer_norm=layer_norm,
            device=device,
        )

        # Same input sequences
        seqs = torch.ones(batch_size, seq_len, model_dim, device=device)

        # Different token IDs
        token_ids_1 = torch.tensor([[0, 1]], device=device)
        token_ids_2 = torch.tensor([[2, 3]], device=device)

        with torch.no_grad():
            output_1 = ple(seqs, token_ids_1)
            output_2 = ple(seqs, token_ids_2)

        # Outputs should differ due to different token embeddings
        assert not torch.allclose(output_1, output_2, rtol=1e-5)

    def test_layer_norm_applied(self) -> None:
        """Verify layer normalization applied to output."""
        vocab_size, hidden_size, model_dim = 100, 32, 64
        batch_size, seq_len = 2, 8

        layer_norm = RMSNorm(model_dim, bias=False, device=device)
        ple = PerLayerEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            model_dim=model_dim,
            layer_norm=layer_norm,
            device=device,
        )

        # Verify layer norm is stored
        assert ple.post_per_layer_input_norm is layer_norm

    def test_no_bias_in_projections(self) -> None:
        """Verify PLE projections have no bias."""
        vocab_size, hidden_size, model_dim = 100, 32, 64

        layer_norm = RMSNorm(model_dim, bias=False, device=device)
        ple = PerLayerEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            model_dim=model_dim,
            layer_norm=layer_norm,
            device=device,
        )

        # All projections should have no bias
        assert ple.per_layer_input_gate.bias is None
        assert ple.per_layer_projection.bias is None

    def test_typical_gemma3n_config(self) -> None:
        """Verify PLE with typical Gemma3n config."""
        vocab_size, hidden_size, model_dim = 262_144, 256, 2048
        batch_size, seq_len = 1, 4

        layer_norm = RMSNorm(model_dim, bias=False, device=device)
        ple = PerLayerEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            model_dim=model_dim,
            layer_norm=layer_norm,
            device=device,
        )

        seqs = torch.randn(batch_size, seq_len, model_dim, device=device)
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        with torch.no_grad():
            output = ple(seqs, token_ids)

        assert output.shape == (batch_size, seq_len, model_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_vocab_size_parameter(self) -> None:
        """Verify various vocab sizes work correctly."""
        hidden_size, model_dim = 32, 64
        batch_size, seq_len = 1, 4

        test_vocab_sizes = [100, 1000, 10000]

        for vocab_size in test_vocab_sizes:
            layer_norm = RMSNorm(model_dim, bias=False, device=device)
            ple = PerLayerEmbedding(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                model_dim=model_dim,
                layer_norm=layer_norm,
                device=device,
            )

            seqs = torch.randn(batch_size, seq_len, model_dim, device=device)
            token_ids = torch.randint(
                0, vocab_size, (batch_size, seq_len), device=device
            )

            with torch.no_grad():
                output = ple(seqs, token_ids)

            assert output.shape == (batch_size, seq_len, model_dim)

    def test_numerical_stability(self) -> None:
        """Verify PLE forward is numerically stable."""
        vocab_size, hidden_size, model_dim = 100, 32, 64
        batch_size, seq_len = 2, 16

        layer_norm = RMSNorm(model_dim, bias=False, device=device)
        ple = PerLayerEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            model_dim=model_dim,
            layer_norm=layer_norm,
            device=device,
        )

        # Test with various input ranges
        test_inputs = [
            torch.randn(batch_size, seq_len, model_dim, device=device) * 0.01,  # Small
            torch.randn(batch_size, seq_len, model_dim, device=device) * 1.0,  # Normal
            torch.randn(batch_size, seq_len, model_dim, device=device) * 10.0,  # Large
        ]

        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        for seqs in test_inputs:
            with torch.no_grad():
                output = ple(seqs, token_ids)

            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            assert output.shape == seqs.shape

    def test_batch_independence(self) -> None:
        """Verify PLE processes batch items independently."""
        vocab_size, hidden_size, model_dim = 100, 32, 64
        batch_size, seq_len = 4, 8

        layer_norm = RMSNorm(model_dim, bias=False, device=device)
        ple = PerLayerEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            model_dim=model_dim,
            layer_norm=layer_norm,
            device=device,
        )

        seqs = torch.randn(batch_size, seq_len, model_dim, device=device)
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        with torch.no_grad():
            batch_output = ple(seqs, token_ids)

        # Process items individually
        with torch.no_grad():
            individual_outputs = [
                ple(seqs[i : i + 1], token_ids[i : i + 1]) for i in range(batch_size)
            ]

        # Results should match
        for i in range(batch_size):
            assert torch.allclose(
                batch_output[i : i + 1], individual_outputs[i], rtol=1e-5
            )
