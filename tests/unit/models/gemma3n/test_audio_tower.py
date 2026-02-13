# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq2.models.gemma3n.audio_tower import Gemma3nAudioTower
from fairseq2.models.gemma3n.config import Gemma3nAudioConfig, get_gemma3n_e2b_config


class TestGemma3nAudioTower:
    def test_end_to_end_shape(self) -> None:
        """Test that full audio tower produces correct output shape."""
        audio_config = Gemma3nAudioConfig()
        text_config = get_gemma3n_e2b_config()

        tower = Gemma3nAudioTower(audio_config, text_config)

        batch_size = 2
        time_steps = 100
        mel_features = torch.randn(batch_size, time_steps, audio_config.input_feat_size)

        output, layout = tower(mel_features)

        # Verify 4x downsampling in time
        expected_time = time_steps // 4
        assert output.shape == (batch_size, expected_time, text_config.model_dim)

        # Verify layout
        assert len(layout.seq_lens) == batch_size
        assert layout.seq_lens == [expected_time] * batch_size

    def test_4x_downsampling(self) -> None:
        """Test that 4x temporal downsampling is applied correctly."""
        audio_config = Gemma3nAudioConfig()
        text_config = get_gemma3n_e2b_config()

        tower = Gemma3nAudioTower(audio_config, text_config)

        batch_size = 1
        for time_steps in [40, 80, 120, 160]:
            mel_features = torch.randn(batch_size, time_steps, audio_config.input_feat_size)
            output, layout = tower(mel_features)

            expected_time = time_steps // 4
            assert output.size(1) == expected_time, \
                f"Input {time_steps} → Expected {expected_time}, got {output.size(1)}"
            assert layout.seq_lens[0] == expected_time

    def test_projection_to_text_space(self) -> None:
        """Test that audio features are projected to text model dimension."""
        audio_config = Gemma3nAudioConfig()
        text_config = get_gemma3n_e2b_config()

        tower = Gemma3nAudioTower(audio_config, text_config)

        batch_size = 2
        time_steps = 100
        mel_features = torch.randn(batch_size, time_steps, audio_config.input_feat_size)

        output, _ = tower(mel_features)

        # Output should be in text model space (2048 for E2B)
        assert output.size(-1) == text_config.model_dim
        assert output.size(-1) == 2048

    def test_different_batch_sizes(self) -> None:
        """Test that tower handles different batch sizes correctly."""
        audio_config = Gemma3nAudioConfig()
        text_config = get_gemma3n_e2b_config()

        tower = Gemma3nAudioTower(audio_config, text_config)

        time_steps = 100

        for batch_size in [1, 2, 4, 8]:
            mel_features = torch.randn(batch_size, time_steps, audio_config.input_feat_size)
            output, layout = tower(mel_features)

            expected_time = time_steps // 4
            assert output.shape == (batch_size, expected_time, text_config.model_dim)
            assert len(layout.seq_lens) == batch_size

    def test_has_all_components(self) -> None:
        """Test that tower contains all expected components."""
        audio_config = Gemma3nAudioConfig()
        text_config = get_gemma3n_e2b_config()

        tower = Gemma3nAudioTower(audio_config, text_config)

        # Verify all components exist
        assert hasattr(tower, "subsample")
        assert hasattr(tower, "encoder")
        assert hasattr(tower, "embedder")

        # Verify subsample has 10 parameters (conv+norm layers + proj)
        subsample_params = list(tower.subsample.parameters())
        assert len(subsample_params) == 10

        # Verify encoder has 12 layers
        assert len(tower.encoder.layers) == 12

        # Verify embedder has embedding + norms + projection
        embedder_params = list(tower.embedder.named_parameters())
        assert any("embedding.weight" in name for name, _ in embedder_params)
        assert any("embedding_projection.weight" in name for name, _ in embedder_params)

    def test_parameter_count(self) -> None:
        """Test that tower has reasonable number of parameters."""
        audio_config = Gemma3nAudioConfig()
        text_config = get_gemma3n_e2b_config()

        tower = Gemma3nAudioTower(audio_config, text_config)

        total_params = sum(p.numel() for p in tower.parameters())

        # Audio tower should have millions of parameters (conformer is large)
        assert total_params > 1_000_000, f"Expected >1M params, got {total_params:,}"

        print(f"Total audio tower parameters: {total_params:,}")

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through entire tower."""
        audio_config = Gemma3nAudioConfig()
        text_config = get_gemma3n_e2b_config()

        tower = Gemma3nAudioTower(audio_config, text_config)

        batch_size = 2
        time_steps = 100
        mel_features = torch.randn(batch_size, time_steps, audio_config.input_feat_size)

        output, _ = tower(mel_features)

        # Create dummy loss and backprop
        loss = output.sum()
        loss.backward()

        # Check that all components have gradients
        subsample_has_grad = any(p.grad is not None for p in tower.subsample.parameters())
        encoder_has_grad = any(p.grad is not None for p in tower.encoder.parameters())
        embedder_has_grad = any(p.grad is not None for p in tower.embedder.parameters())

        assert subsample_has_grad, "Subsample should have gradients"
        assert encoder_has_grad, "Encoder should have gradients"
        assert embedder_has_grad, "Embedder should have gradients"
